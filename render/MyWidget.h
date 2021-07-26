#ifndef MY_WIDGET_H_
#define MY_WIDGET_H_

#include "dart/gui/osg/ImGuiViewer.hpp"
#include "dart/gui/osg/ImGuiWidget.hpp"

class MyWorldNode;

class MyWidget : public dart::gui::osg::ImGuiWidget
{
public:
  /// Constructor
  MyWidget(
      dart::gui::osg::ImGuiViewer* viewer, MyWorldNode* node);

  // Documentation inherited
  void render() override;

protected:
//   void setGravity(float gravity);

  ::osg::ref_ptr<dart::gui::osg::ImGuiViewer> mViewer;

  ::osg::ref_ptr<MyWorldNode> mNode;

//   float mGuiGravityAcc;

//   float mGravityAcc;

//   bool mGuiHeadlights;

//   bool mShadow;

  /// Control mode value for GUI
  int mGuiControlMode;

  /// Actual control mode
  ///   - 0: No control
  ///   - 1: Short-stride walking control
  ///   - 1: Normal-stride walking control
  int mControlMode;
};

#endif // DART_EXAMPLE_OSG_OSGATLASSIMBICON_ATLASSIMBICONWIDGET_HPP_
