// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#include <librealsense2/rs.hpp>     // Include RealSense Cross Platform API
#include "example.hpp"              // Include short list of convenience functions for rendering

#include <string>
#include <map>
#include <algorithm>
#include <mutex>                    // std::mutex, std::lock_guard
#include <cmath>                    // std::ceil
#include <sstream>
#include <iostream>
#include <fstream>
#include <cstring>

const std::string no_camera_message = "No camera connected, please connect 1 or more";
const std::string platform_camera_name = "Platform Camera";

class device_container
{
    // Helper struct per pipeline
    struct view_port
    {
        std::map<int, rs2::frame> frames_per_stream;
        rs2::colorizer colorize_frame;
        texture tex;
        rs2::pipeline pipe;
        rs2::pipeline_profile profile;
    };

public:

    void enable_device(rs2::device dev)
    {
        std::string serial_number(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
        std::lock_guard<std::mutex> lock(_mutex);

        if (_devices.find(serial_number) != _devices.end())
        {
            return; //already in
        }

        // Ignoring platform cameras (webcams, etc..)
        if (platform_camera_name == dev.get_info(RS2_CAMERA_INFO_NAME))
        {
            return;
        }
        // Create a pipeline from the given device
        rs2::pipeline p;
        rs2::config c;
        c.enable_device(serial_number);
        c.disable_all_streams();
        c.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGB8, 15);
        c.enable_stream(RS2_STREAM_DEPTH, 0, 1280, 720, RS2_FORMAT_Z16, 15);
        // Start the pipeline with the configuration
        rs2::pipeline_profile profile = p.start(c);
        // Hold it internally
        _devices.emplace(serial_number, view_port{ {},{},{}, p, profile });

    }

    void remove_devices(const rs2::event_information& info)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        // Go over the list of devices and check if it was disconnected
        auto itr = _devices.begin();
        while(itr != _devices.end())
        {
            if (info.was_removed(itr->second.profile.get_device()))
            {
                itr = _devices.erase(itr);
            }
            else
            {
                ++itr;
            }
        }
    }

    size_t device_count()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _devices.size();
    }

    int stream_count()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        int count = 0;
        for (auto&& sn_to_dev : _devices)
        {
            for (auto&& stream : sn_to_dev.second.frames_per_stream)
            {
                if (stream.second)
                {
                    count++;
                }
            }
        }
        return count;
    }

    void poll_frames()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        // Go over all device
        for (auto&& view : _devices)
        {
            // Ask each pipeline if there are new frames available
            rs2::frameset frameset;
            if (view.second.pipe.poll_for_frames(&frameset))
            {
                for (int i = 0; i < frameset.size(); i++)
                {
                    rs2::frame new_frame = frameset[i];
                    int stream_id = new_frame.get_profile().unique_id();

                    rs2_stream align_to = find_stream_to_align(view.second.profile.get_streams());
                    rs2::align align(align_to);
                    float depth_scale = get_depth_scale(view.second.profile.get_device());

                    if (profile_changed(view.second.pipe.get_active_profile().get_streams(), view.second.profile.get_streams()))
                    {
                        //If the profile was changed, update the align object, and also get the new device's depth scale
                        align_to = find_stream_to_align(view.second.profile.get_streams());
                        align = rs2::align(align_to);
                        depth_scale = get_depth_scale(view.second.profile.get_device());
                    }

                    //Get processed aligned frame
                    auto processed = align.process(frameset);

                    // Trying to get both other and aligned depth frames
                    rs2::video_frame other_frame = processed.first(align_to);
                    rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();

                    //If one of them is unavailable, continue iteration
                    if (!aligned_depth_frame || !other_frame)
                    {
                        continue;
                    }
                    // Passing both frames to remove_background so it will "strip" the background
                    // NOTE: in this example, we alter the buffer of the other frame, instead of copying it and altering the copy
                    //       This behavior is not recommended in real application since the other frame could be used elsewhere
                    float depth_clipping_distance = 5.f;
                    remove_background(other_frame, aligned_depth_frame, depth_scale, depth_clipping_distance);

                    view.second.frames_per_stream[stream_id] = view.second.colorize_frame(new_frame); //update view port with the new stream
                }
            }
        }
    }

    bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
    {
        for (auto&& sp : prev)
        {
            //If previous profile is in current (maybe just added another)
            auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
            if (itr == std::end(current)) //If it previous stream wasn't found in current
            {
                return true;
            }
        }
        return false;
    }

    void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
    {
        const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
        uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

        int width = other_frame.get_width();
        int height = other_frame.get_height();
        int other_bpp = other_frame.get_bytes_per_pixel();

#pragma omp parallel for schedule(dynamic) //Using OpenMP to try to parallelise the loop
        for (int y = 0; y < height; y++)
        {
            auto depth_pixel_index = y * width;
            for (int x = 0; x < width; x++, ++depth_pixel_index)
            {
                // Get the depth value of the current pixel
                auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];

                // Check if the depth value is invalid (<=0) or greater than the threashold
                if (pixels_distance <= 0.f || pixels_distance > clipping_dist)
                {
                    // Calculate the offset in other frame's buffer to current pixel
                    auto offset = depth_pixel_index * other_bpp;

                    // Set pixel to "background" color (0x999999)
                    std::memset(&p_other_frame[offset], 0x99, other_bpp);
                }
            }
        }
    }

    rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
    {
        //Given a vector of streams, we try to find a depth stream and another stream to align depth with.
        //We prioritize color streams to make the view look better.
        //If color is not available, we take another stream that (other than depth)
        rs2_stream align_to = RS2_STREAM_ANY;
        bool depth_stream_found = false;
        bool color_stream_found = false;
        for (rs2::stream_profile sp : streams)
        {
            rs2_stream profile_stream = sp.stream_type();
            if (profile_stream != RS2_STREAM_DEPTH)
            {
                if (!color_stream_found)         //Prefer color
                    align_to = profile_stream;

                if (profile_stream == RS2_STREAM_COLOR)
                {
                    color_stream_found = true;
                }
            }
            else
            {
                depth_stream_found = true;
            }
        }

        if(!depth_stream_found)
            throw std::runtime_error("No Depth stream available");

        if (align_to == RS2_STREAM_ANY)
            throw std::runtime_error("No stream found to align with Depth");

        return align_to;
    }

    float get_depth_scale(rs2::device dev)
    {
        // Go over the device's sensors
        for (rs2::sensor& sensor : dev.query_sensors())
        {
            // Check if the sensor if a depth sensor
            if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
            {
                return dpt.get_depth_scale();
            }
        }
        throw std::runtime_error("Device does not have a depth sensor");
    }

    void render_textures(int cols, int rows, float view_width, float view_height)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        int stream_no = 0;
        for (auto&& view : _devices)
        {
            // For each device get its frames
            for (auto&& id_to_frame : view.second.frames_per_stream)
            {
                // If the frame is available
                if (id_to_frame.second)
                {
                    view.second.tex.upload(id_to_frame.second);
                }
                rect frame_location{ view_width * (stream_no % cols), view_height * (stream_no / cols), view_width, view_height };
                if (rs2::video_frame vid_frame = id_to_frame.second.as<rs2::video_frame>())
                {
                    rect adjuested = frame_location.adjust_ratio({ static_cast<float>(vid_frame.get_width())
                                                                         , static_cast<float>(vid_frame.get_height()) });
                    view.second.tex.show(adjuested);
                    stream_no++;
                }
            }
        }
    }
private:
    std::mutex _mutex;
    std::map<std::string, view_port> _devices;
};


int main(int argc, char * argv[]) try
{
    // Create a simple OpenGL window for rendering:
    window app(1280, 960, "CPP Multi-Camera Example");

    device_container connected_devices;

    rs2::context ctx;    // Create librealsense context for managing devices

    // Register callback for tracking which devices are currently connected
    ctx.set_devices_changed_callback([&](rs2::event_information& info)
                                     {
                                         connected_devices.remove_devices(info);
                                         for (auto&& dev : info.get_new_devices())
                                         {
                                             connected_devices.enable_device(dev);
                                         }
                                     });

    // Initial population of the device list
    for (auto&& dev : ctx.query_devices()) // Query the list of connected RealSense devices
    {
        connected_devices.enable_device(dev);
    }

    while (app) // Application still alive?
    {
        connected_devices.poll_frames();
        auto total_number_of_streams = connected_devices.stream_count();
        if (total_number_of_streams == 0)
        {
            draw_text(int(std::max(0.f, (app.width() / 2) - no_camera_message.length() * 3)),
                      int(app.height() / 2), no_camera_message.c_str());
            continue;
        }
        if (connected_devices.device_count() == 1)
        {
            draw_text(0, 10, "Please connect another camera");
        }
        int cols = int(std::ceil(std::sqrt(total_number_of_streams)));
        int rows = int(std::ceil(total_number_of_streams / static_cast<float>(cols)));

        float view_width = (app.width() / cols);
        float view_height = (app.height() / rows);

        connected_devices.render_textures(cols, rows, view_width, view_height);
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception & e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
