#include "camera.hpp"

Camera::Camera(){}

void Camera::process_keyboard_input(GLFWwindow * window, float delta_time){
    float camera_speed = m_speed_mul * delta_time;
    if(glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS){
        camera_speed = m_faster_speed_mul * delta_time;
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS){
        m_pos += camera_speed * m_front;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS){
        m_pos -= camera_speed * m_front;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS){
        m_pos -= glm::normalize(glm::cross(m_front, m_up)) * camera_speed;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS){
        m_pos += glm::normalize(glm::cross(m_front, m_up)) * camera_speed;
    }
}

void Camera::process_mouse_input(GLFWwindow * window, double xpos, double ypos){

    if(ImGui::GetIO().WantCaptureMouse) return; // let ImGui have mouse input

    if(glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) != GLFW_PRESS){
        m_first_mouse = true;
        return;
    }
    
    // Prevents pop in jerk on first mouse
    if (m_first_mouse)
    {
        m_last_x = xpos;
        m_last_y = ypos;
        m_first_mouse = false;
    }
    float xoffset = xpos - m_last_x;
    float yoffset = m_last_y - ypos; // reversed since y-coordinates range from bottom to top
    m_last_x = xpos;
    m_last_y = ypos;

    const float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    m_yaw += xoffset;
    m_pitch += yoffset;
    if(m_pitch > 89.0f){
        m_pitch =  89.0f;
    }
    if(m_pitch < -89.0f){
        m_pitch = -89.0f;
    }

    glm::vec3 direction;
    direction.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    direction.y = sin(glm::radians(m_pitch));
    direction.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    m_front = glm::normalize(direction);
}

glm::mat4 Camera::get_view_matrix(){
    return glm::lookAt(m_pos, m_pos + m_front, m_up);
}

glm::vec3 Camera::get_camera_pos(){
    return m_pos;
}
glm::vec3 Camera::get_camera_front(){
    return m_front;
}
glm::vec3 Camera::get_camera_up(){
    return m_up;
}
