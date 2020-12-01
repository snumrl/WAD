#ifndef __MASS_DRAW_UTILS_H__
#define __MASS_DRAW_UTILS_H__

#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"

namespace MASS
{
namespace DrawUtils
{
    void DrawGLBegin();
    void DrawGLEnd();
    void DrawBaseGraph(double x, double y, double w, double h, double ratio, double offset, std::string name);

    void DrawQuads(double x, double y, double w, double h, Eigen::Vector4d color);

    void DrawString(double x, double y, std::string str);
    void DrawString(double x, double y, std::string str, Eigen::Vector4d color);
    void DrawStringMean(double x, double y, double w, double h, double ratio, double offset_x, double offset_y, double offset, std::vector<double> data, Eigen::Vector4d color);
    void DrawStringMean(double x, double y, double w, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color);
    void DrawStringMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::vector<double> data, Eigen::Vector4d color);
    void DrawStringMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color);
    void DrawStringMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color, double lower, double upper);
    void DrawStringMinMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color);

    void DrawLine(double p1_x, double p1_y, double p2_x, double p2_y, Eigen::Vector4d color, double line_width);
    void DrawLineStipple(double p1_x, double p1_y, double p2_x, double p2_y, Eigen::Vector4d color, double line_width);
    void DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::vector<double>& data);
    void DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::vector<double>& data, Eigen::Vector4d color1, double line_width1, std::vector<double>& data1);
    void DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data);
    void DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, double lower, double upper);
    void DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, int idx);
    void DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, Eigen::Vector4d color1, double line_width1, std::deque<double>& data1);
    void DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, int idx, Eigen::Vector4d color1, double line_width1, std::deque<double>& data1, int idx1);
}
}
#endif
