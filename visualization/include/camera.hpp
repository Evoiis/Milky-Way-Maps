#ifndef CAM_H
#define CAM_H

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "imgui.h"

class Camera{
public:

    Camera();

    void process_keyboard_input(GLFWwindow * window, float delta_time);
    void process_mouse_input(GLFWwindow * window, double xpos, double ypos);

    glm::mat4 get_view_matrix();
    glm::vec3 get_camera_pos();
    glm::vec3 get_camera_front();
    glm::vec3 get_camera_up();

private:
    glm::vec3 m_pos   = glm::vec3(0.0f, 0.0f,  3.0f);
    glm::vec3 m_front = glm::vec3(0.0f, 0.0f, -1.0f); // direction camera is pointing
    glm::vec3 m_up    = glm::vec3(0.0f, 1.0f,  0.0f); // which way is up for the camera, can edit to roll the camera
    
    float m_speed_mul = 10.f;
    float m_faster_speed_mul = 40.f;

    bool m_first_mouse;
    float m_last_x;
    float m_last_y;

    float m_yaw = -90.f;
    float m_pitch = 0.f;
};

#endif

