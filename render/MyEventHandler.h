#ifndef MY_EVENT_HANDLER_H_
#define MY_EVENT_HANDLER_H_

#include <dart/dart.hpp>
#include <dart/gui/osg/osg.hpp>
#include <dart/utils/utils.hpp>

#include "MyWorldNode.h"

class MyEventHandler : public osgGA::GUIEventHandler
{
public:
  MyEventHandler(MyWorldNode* node);

  bool handle(
      const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter&) override;

protected:
  ::osg::ref_ptr<MyWorldNode> mNode;
};

#endif 
