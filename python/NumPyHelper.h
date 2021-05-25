#ifndef __NUMPY_HELPER_H__
#define __NUMPY_HELPER_H__
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <string>
#include <map>

namespace py = pybind11;

py::array_t<float> toNumPyArray(const std::vector<bool>& val);
py::array_t<float> toNumPyArray(const std::vector<float>& val);
py::array_t<float> toNumPyArray(const std::vector<double>& val);
py::array_t<float> toNumPyArray(const std::pair<double,double>& val);
py::array_t<float> toNumPyArray(const std::vector<Eigen::VectorXd>& val);
py::array_t<float> toNumPyArray(const std::vector<Eigen::MatrixXd>& val);
py::array_t<float> toNumPyArray(const std::vector<std::vector<float>>& val);
py::array_t<float> toNumPyArray(const std::vector<std::vector<double>>& val);
py::array_t<float> toNumPyArray(const Eigen::VectorXd& vec);
py::array_t<float> toNumPyArray(const Eigen::MatrixXd& matrix);
py::array_t<float> toNumPyArray(const Eigen::Isometry3d& T);

std::vector<bool> toStdVector(const py::list& list);
Eigen::VectorXd toEigenVector(const py::array_t<float>& array);
Eigen::MatrixXd toEigenMatrix(const py::array_t<float>& array);
std::vector<Eigen::VectorXd> toEigenVectorVector(const py::array_t<float>& array);

// py::dict toPythonDict(const std::map<std::string, double>& map);
// p::dict toPythonDict(const std::map<std::string, double>& map);
#endif
