#include "MyWidget.h"
#include "MyWorldNode.h"
#include "dart/external/imgui/imgui.h"

//==============================================================================
// MyWidget::MyWidget(
//     dart::gui::osg::ImGuiViewer* viewer, AtlasSimbiconWorldNode* node)
//   : mViewer(viewer),
//     mNode(node),
//     mGuiGravityAcc(9.81f),
//     mGravityAcc(mGuiGravityAcc),
//     mGuiHeadlights(true),
//     mGuiControlMode(2),
//     mControlMode(2)
// {
//   // Do nothing
// }

MyWidget::MyWidget(
    dart::gui::osg::ImGuiViewer* viewer, MyWorldNode* node)
  : mViewer(viewer),
    mNode(node),
    mGuiControlMode(2),
    mControlMode(2)
{
  // Do nothing
}

//==============================================================================
void MyWidget::render()
{
  ImGui::SetNextWindowPos(ImVec2(10, 20));
  ImGui::SetNextWindowSize(ImVec2(360, 400));
  ImGui::SetNextWindowBgAlpha(0.5f);
  if (!ImGui::Begin(
          "Atlas Control",
          nullptr,
          ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar))
  {
    // Early out if the window is collapsed, as an optimization.
    ImGui::End();
    return;
  }

  // Menu
  if (ImGui::BeginMenuBar())
  {
    if (ImGui::BeginMenu("Menu"))
    {
      if (ImGui::MenuItem("Exit"))
        mViewer->setDone(true);
      ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help"))
    {
      if (ImGui::MenuItem("About DART"))
        mViewer->showAbout();
      ImGui::EndMenu();
    }
    ImGui::EndMenuBar();
  }

  ImGui::Text("Altas robot controlled by Simbicon");
  ImGui::Spacing();

  if (ImGui::CollapsingHeader("Help"))
  {
    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + 320);
    ImGui::Text("User Guid:\n");
    ImGui::Text("%s", mViewer->getInstructions().c_str());
    ImGui::Text("Press [r] to reset Atlas to the initial position.\n");
    ImGui::Text("Press [a] to push forward Atlas toroso.\n");
    ImGui::Text("Press [s] to push backward Atlas toroso.\n");
    ImGui::Text("Press [d] to push left Atlas toroso.\n");
    ImGui::Text("Press [f] to push right Atlas toroso.\n");
    ImGui::Text("Left-click on a block to select it.\n");
    ImGui::PopTextWrapPos();
  }

  if (ImGui::CollapsingHeader("Simulation", ImGuiTreeNodeFlags_DefaultOpen))
  {
    int e = mViewer->isSimulating() ? 0 : 1;
    if (mViewer->isAllowingSimulation())
    {
      if (ImGui::RadioButton("Play", &e, 0) && !mViewer->isSimulating())
        mViewer->simulate(true);
      ImGui::SameLine();
      if (ImGui::RadioButton("Pause", &e, 1) && mViewer->isSimulating())
        mViewer->simulate(false);
    }
  }

//   if (ImGui::CollapsingHeader("World Options", ImGuiTreeNodeFlags_DefaultOpen))
//   {
//     // Gravity
//     ImGui::SliderFloat("Gravity Acc.", &mGuiGravityAcc, 5.0, 20.0, "-%.2f");
//     setGravity(mGuiGravityAcc);

//     ImGui::Spacing();

//     // Headlights
//     mGuiHeadlights = mViewer->checkHeadlights();
//     if (ImGui::Checkbox("Headlights On/Off", &mGuiHeadlights))
//     {
//       mViewer->switchHeadlights(mGuiHeadlights);
//     }

//     // Shadow
//     mShadow = mNode->isShadowed();
//     if (ImGui::Checkbox("Shadow On/Off", &mShadow))
//     {
//       if (mShadow)
//         mNode->showShadow();
//       else
//         mNode->hideShadow();
//     }
//   }

  if (ImGui::CollapsingHeader(
          "Atlas Simbicon Options", ImGuiTreeNodeFlags_DefaultOpen))
  {
    const auto reset = ImGui::Button("Reset Atlas");
    // if (reset)
    //   mNode->reset();

    ImGui::Spacing();

    // Stride
    ImGui::RadioButton("No Control", &mGuiControlMode, 0);
    ImGui::RadioButton("Short-Stride Walking", &mGuiControlMode, 1);
    ImGui::RadioButton("Normal-Stride Walking", &mGuiControlMode, 2);

    if (mGuiControlMode != mControlMode)
    {
      switch (mGuiControlMode)
      {
        case 0:
        std::cout << "case0" << std::endl;
        //   mNode->switchToNoControl();
          break;
        case 1:
        std::cout << "case1" << std::endl;
        //   mNode->switchToShortStrideWalking();
          break;
        case 2:
        std::cout << "case2" << std::endl;
        //   mNode->switchToNormalStrideWalking();
          break;
      }

      mControlMode = mGuiControlMode;
    }
  }

  ImGui::End();
}

//==============================================================================
// void MyWidget::setGravity(float gravity)
// {
//   if (mGravityAcc == gravity)
//     return;

//   mGravityAcc = gravity;
//   mNode->getWorld()->setGravity(-mGravityAcc * Eigen::Vector3d::UnitY());
// }
