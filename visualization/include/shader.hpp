#ifndef SHADER_HPP
#define SHADER_HPP

#include <GL/glew.h> 

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader{
public:
    unsigned int m_ID;
    Shader(const char* vertexPath, const char* fragmentPath);
    
    void use();
    
    // void setBool(const std::string &name, bool value) const;
    
    // void setInt(const std::string &name, int value) const;    
    
    // void setFloat(const std::string &name, float value) const;

private:
    // utility function for checking shader compilation/linking errors.
    
    void checkCompileErrors(unsigned int shader, std::string type);
};
#endif

