#include "MyEventHandler.h"

//==============================================================================
MyEventHandler::MyEventHandler(
    MyWorldNode* node)
  : mNode(node)
{
  // Do nothing
}

//==============================================================================
bool MyEventHandler::handle(
    const osgGA::GUIEventAdapter& ea, osgGA::GUIActionAdapter&)
{
  if (ea.getEventType() == osgGA::GUIEventAdapter::KEYDOWN)
  {
    if (ea.getKey() == 'r' || ea.getKey() == 'R')
    {

        mNode->Reset();
        std::cout << "press 'r' or 'R'" << std::endl;
        return true;
    }
    else if (ea.getKey() == 'a' || ea.getKey() == 'A')
    {
      
        return true;
    }
    else if (ea.getKey() == 's' || ea.getKey() == 'S')
    {
        mNode->Step();
        return true;
    }
    else if (ea.getKey() == 'd' || ea.getKey() == 'D')
    {
        return true;
    }
    else if (ea.getKey() == 'f' || ea.getKey() == 'F')
    {
        return true;
    }
  }

  // The return value should be 'true' if the input has been fully handled
  // and should not be visible to any remaining event handlers. It should be
  // false if the input has not been fully handled and should be viewed by
  // any remaining event handlers.
  return false;
}
