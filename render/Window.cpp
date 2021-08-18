#include "Window.h"
#include <iostream>
#include <ctime>

Eigen::Vector4d white(0.9, 0.9, 0.9, 1.0);
Eigen::Vector4d black(0.1, 0.1, 0.1, 1.0);
Eigen::Vector4d grey(0.6, 0.6, 0.6, 1.0);
Eigen::Vector4d red(0.8, 0.2, 0.2, 1.0);
Eigen::Vector4d green(0.2, 0.8, 0.2, 1.0);
Eigen::Vector4d blue(0.2, 0.2, 0.8, 1.0);
Eigen::Vector4d yellow(0.8, 0.8, 0.2, 1.0);
Eigen::Vector4d purple(0.8, 0.2, 0.8, 1.0);

Eigen::Vector4d white_trans(0.9, 0.9, 0.9, 0.2);
Eigen::Vector4d black_trans(0.1, 0.1, 0.1, 0.2);
Eigen::Vector4d grey_trans(0.6, 0.6, 0.6, 0.2);
Eigen::Vector4d red_trans(0.8, 0.2, 0.2, 0.2);
Eigen::Vector4d green_trans(0.2, 0.8, 0.2, 0.2);
Eigen::Vector4d blue_trans(0.2, 0.2, 0.8, 0.2);
Eigen::Vector4d yellow_trans(0.8, 0.8, 0.2, 0.2);
Eigen::Vector4d purple_trans(0.8, 0.2, 0.8, 0.2);

namespace WAD
{

Window::
Window(Environment* env)
	:mEnv(env),mFocus(true),mSimulating(false),mDrawCharacter(true),mDrawTarget(false),mDrawReference(false),mDrawOBJ(false),mDrawShadow(true),mMuscleNNLoaded(false),mDeviceNNLoaded(false),mDevice_On(false),isDrawCharacter(false),isDrawTarget(false),isDrawReference(false),isDrawDevice(false),mDrawArrow(false),mDrawGraph(false),mMetabolicEnergyMode(0),mJointTorqueMode(0),mJointAngleMode(0),mCharacterMode(0),mParamMode(0),mViewMode(0),mDrawParameter(true),mTalusL(false),mTalusR(false), mDisplayIter(0)
{
	mBackground[0] = 0.96;
	mBackground[1] = 0.96;
	mBackground[2] = 0.97;
	mBackground[3] = 0.7;
	SetFocus();
	mZoom = 0.30;
	mNNLoaded = false;
	mDevice_On = env->GetCharacter()->GetDevice_OnOff();
	mFootinterval.resize(20);
	mDisplayTimeout = 1000 / 60.0;

	mm = py::module::import("__main__");
	mns = mm.attr("__dict__");
	
	py::str module_dir = (std::string(WAD_ROOT_DIR)+"/python").c_str();
	sys_module = py::module::import("sys");
	sys_module.attr("path").attr("insert")(1, module_dir);
	
	py::exec("import torch",mns);
	py::exec("import torch.nn as nn",mns);
	py::exec("import torch.optim as optim",mns);
	py::exec("import torch.nn.functional as F",mns);
	py::exec("import numpy as np",mns);
	py::exec("from Model import *",mns);
	py::exec("from RunningMeanStd import *",mns);
}

Window::
Window(Environment* env, const std::string& nn_path)
	:Window(env)
{
	mNNLoaded = true;

	py::str str;
	str = ("num_state = "+std::to_string(mEnv->GetCharacter()->GetNumState())).c_str();
	py::exec(str, mns);
	str = ("num_action = "+std::to_string(mEnv->GetCharacter()->GetNumAction())).c_str();
	py::exec(str, mns);

	nn_module = py::eval("SimulationNN(num_state,num_action)", mns);
	py::object load = nn_module.attr("load");
	load(nn_path);
	
	rms_module = py::eval("RunningMeanStd()", mns);
	py::object load_rms = rms_module.attr("load2");
	load_rms(nn_path);

}

Window::
Window(Environment* env,const std::string& nn_path, const std::string& nn_path2)
	:Window(env, nn_path)
{
	if(env->GetUseMuscle())
		LoadMuscleNN(nn_path2);

	if(env->GetUseDeviceNN())
		LoadDeviceNN(nn_path2);
}

Window::
Window(Environment* env,const std::string& nn_path,const std::string& muscle_nn_path, const std::string& device_nn_path)
	:Window(env, nn_path)
{
	if(env->GetUseMuscle())
		LoadMuscleNN(muscle_nn_path);

	if(env->GetUseDevice())
		LoadDeviceNN(device_nn_path);
}

void
Window::
LoadMuscleNN(const std::string& muscle_nn_path)
{
	mMuscleNNLoaded = true;

	py::str str;
	str = ("num_total_muscle_related_dofs = "+std::to_string(mEnv->GetCharacter()->GetNumTotalRelatedDofs())).c_str();
	py::exec(str,mns);
	str = ("num_actions = "+std::to_string(mEnv->GetCharacter()->GetNumActiveDof())).c_str();
	py::exec(str,mns);
	str = ("num_muscles = "+std::to_string(mEnv->GetCharacter()->GetNumMuscles())).c_str();
	py::exec(str,mns);

	mMuscleNum = mEnv->GetCharacter()->GetNumMuscles();
	mMuscleMapNum = mEnv->GetCharacter()->GetNumMusclesMap();

	muscle_nn_module = py::eval("MuscleNN(num_total_muscle_related_dofs,num_actions,num_muscles)",mns);

	py::object load = muscle_nn_module.attr("load");
	load(muscle_nn_path);
}

void
Window::
LoadDeviceNN(const std::string& device_nn_path)
{
	mDeviceNNLoaded = true;
	mDevice_On = true;

	py::str str;
	str = ("num_state_device = "+std::to_string(mEnv->GetCharacter()->GetDevice()->GetNumState())).c_str();
	py::exec(str,mns);
	str = ("num_action_device = "+std::to_string(mEnv->GetCharacter()->GetDevice()->GetNumAction())).c_str();
	py::exec(str,mns);

	device_nn_module = py::eval("SimulationNN(num_state_device,num_action_device)",mns);

	py::object load = device_nn_module.attr("load");
	load(device_nn_path);
}

void
Window::
ParamChange(bool b)
{
	if(!mEnv->GetUseAdaptiveSampling())
		return;
	Eigen::VectorXd min_v = mEnv->GetMinV();
	Eigen::VectorXd max_v = mEnv->GetMaxV();
	if(b)
	{
		if(mParamMode == 1)
		{
			double m_ratio = mEnv->GetCharacter()->GetMassRatio();
			m_ratio += 0.10;
			if(m_ratio > max_v[0])
				m_ratio = max_v[0];
			mEnv->GetCharacter()->SetMassRatio(m_ratio);
		}
		else if(mParamMode == 2)
		{
			double f_ratio = mEnv->GetCharacter()->GetForceRatio();
			f_ratio += 0.05;
			if(f_ratio > max_v[1])
				f_ratio = max_v[1];
			mEnv->GetCharacter()->SetForceRatio(f_ratio);
		}
		else if(mParamMode ==3)
		{
			double s_ratio = mEnv->GetCharacter()->GetSpeedRatio();
			s_ratio += 0.1;
			if(s_ratio > max_v[2])
				s_ratio = max_v[2];
			mEnv->GetCharacter()->SetSpeedRatio(s_ratio);
			// mEnv->GetCharacter()->SetBVHidx(s_ratio);
		}
		else if(mParamMode == 4)
		{
			double k_ = mEnv->GetDevice()->GetK_();
			k_ += 2.0;
			if(k_ > max_v[3]*30.0)
				k_ = max_v[3]*30.0;
			mEnv->GetDevice()->SetK_(k_);
		}
		else if(mParamMode == 5)
		{
			double t_ = mEnv->GetDevice()->GetDelta_t();
			t_ += 0.02;
			if(t_ > max_v[4])
				t_ = max_v[4];
			mEnv->GetDevice()->SetDelta_t(t_);
		}
	}
	else
	{
		if(mParamMode == 1)
		{
			double m_ratio = mEnv->GetCharacter()->GetMassRatio();
			m_ratio -= 0.10;
			if(m_ratio < min_v[0])
				m_ratio = min_v[0];
			mEnv->GetCharacter()->SetMassRatio(m_ratio);
		}
		else if(mParamMode == 2)
		{
			double f_ratio = mEnv->GetCharacter()->GetForceRatio();
			f_ratio -= 0.05;
			if(f_ratio < min_v[1])
				f_ratio = min_v[1];
			mEnv->GetCharacter()->SetForceRatio(f_ratio);
		}
		else if(mParamMode ==3)
		{
			double s_ratio = mEnv->GetCharacter()->GetSpeedRatio();
			s_ratio -= 0.1;
			if(s_ratio < min_v[2])
				s_ratio = min_v[2];
			mEnv->GetCharacter()->SetSpeedRatio(s_ratio);
			// mEnv->GetCharacter()->SetBVHidx(s_ratio);
		}
		else if(mParamMode == 4)
		{
			double k_ = mEnv->GetDevice()->GetK_();
			k_ -= 2.0;
			if(k_ < min_v[3]*30.0)
				k_ = min_v[3]*30.0;
			mEnv->GetDevice()->SetK_(k_);
		}
		else if(mParamMode == 5)
		{
			double t_ = mEnv->GetDevice()->GetDelta_t();
			t_ -= 0.02;
			if(t_ < min_v[4])
				t_ = min_v[4];
			mEnv->GetDevice()->SetDelta_t(t_);
		}
	}
}

void
Window::
m(bool b)
{
	if(b)
		mMuscleMode = (mMuscleMode+1)%mMuscleMapNum;
	else
		mMuscleMode = (mMuscleMode-1)%mMuscleMapNum;

	std::map<std::string, std::vector<Muscle*>> map = mEnv->GetCharacter()->GetMusclesMap();

	int idx = 0;
	std::vector<Muscle*> muscle_vec;
	for(auto iter = map.begin(); iter != map.end() ; iter++)
	{
		if(idx == mMuscleMode)
			break;

		cur_muscle = iter->first;
		muscle_vec = iter->second;
		idx++;
	}
}

void
Window::
WriteMetaE()
{
	if(!isOpenFile)
	{
		mFile << " MetaE";
		mMetabolicEnergy = mEnv->GetCharacter()->GetMetabolicEnergy();
		auto& HOUD06 = mMetabolicEnergy->GetHOUD06_map_deque();
		for(auto iter = HOUD06.begin(); iter != HOUD06.end(); iter++)
			mFile << " " + iter->first;

		auto&  muscles = mEnv->GetCharacter()->GetMuscles();
		for(auto m : muscles)
		{
			mFile << " " + m->GetName();
		}
	}
	else
	{
		mMetabolicEnergy = mEnv->GetCharacter()->GetMetabolicEnergy();
		std::deque<double> HOUD06 = mMetabolicEnergy->GetHOUD06_deque();
		auto& HOUD06_ = mMetabolicEnergy->GetHOUD06_map_deque();
		mFile << " " + std::to_string(HOUD06.at(0));
		for(auto iter = HOUD06_.begin(); iter != HOUD06_.end(); iter++)
			mFile << " " + std::to_string((iter->second).at(0));

		auto&  muscles = mEnv->GetCharacter()->GetMuscles();
		for(auto m : muscles)
		{
			mFile << " " + std::to_string(m->GetForce());
		}
	}
}

void
Window::
WriteJointAngle()
{
	if(!isOpenFile)
	{
		mJointDatas = mEnv->GetCharacter()->GetJointDatas();
		auto& angles = mJointDatas->GetAngles();
		for(auto iter = angles.begin(); iter != angles.end(); iter++)
			mFile << " " + iter->first;
	}
	else
	{
		mJointDatas = mEnv->GetCharacter()->GetJointDatas();
		auto& angles = mJointDatas->GetAngles();
		for(auto iter = angles.begin(); iter != angles.end(); iter++)
			mFile << " " + std::to_string((iter->second).at(0)*180.0/M_PI);
	}
}

void
Window::
WriteJointTorque()
{
	if(!isOpenFile)
	{
		// mJointDatas = mEnv->GetCharacter()->GetJointDatas();
		// auto& torquesNorm = mJointDatas->GetTorquesNorm();
		// for(auto iter = torquesNorm.begin(); iter != torquesNorm.end(); iter++)
		// 	mFile << " " + iter->first;
	}
	else
	{
		// mJointDatas = mEnv->GetCharacter()->GetJointDatas();
		// auto& torquesNorm = mJointDatas->GetTorquesNorm();
		// for(auto iter = torquesNorm.begin(); iter != torquesNorm.end(); iter++)
		// 	mFile << " " + std::to_string((iter->second).at(0));
	}
}

void
Window::
WriteFileName()
{
	time_t now = std::time(0);
	tm* ltm = std::localtime(&now);

	std::string month = std::to_string(ltm->tm_mon);
	if(ltm->tm_mon < 10)
		month = "0"+month;

	std::string day = std::to_string(ltm->tm_mday);
	if(ltm->tm_mday < 10)
		day = "0"+day;

	std::string hour = std::to_string(ltm->tm_hour);
	if(ltm->tm_hour < 10)
		hour = "0" + hour;

	std::string min = std::to_string(ltm->tm_min);
	if(ltm->tm_min < 10)
		min = "0" + min;

	mFileName = month + day + "_" + hour + min;
}

void
Window::
Write()
{
	if(mWriteFile)
	{
		if(!isOpenFile)
		{
			this->WriteFileName();

			mFile.open(mFileName);
			if(mFile.is_open())
				std::cout << mFileName << " open" << std::endl;

			mFile << "Time Frame AdtFrame";
			// this->WriteMetaE();
			// this->WriteJointAngle();
			this->WriteJointTorque();

			isOpenFile = true;
		}

		if(isOpenFile)
		{
			double t = (mEnv->GetWorld()->getTime());
			double f = mEnv->GetCharacter()->GetFrame();
			double af = mEnv->GetCharacter()->GetAdaptiveFrame();
			mFile << "\n";
			mFile << std::to_string(t) + " " + std::to_string(f) + " " + std::to_string(af);

			// this->WriteMetaE();
			// this->WriteJointAngle();
			this->WriteJointTorque();
		}
	}
	else
	{
		if(isOpenFile)
		{
			mFile.close();
			std::cout << mFileName << " close" << std::endl;

			// mFileName += "_avg";
			// mFile.open(mFileName);
			// if(mFile.is_open())
			// 	std::cout << mFileName << " open" << std::endl;

			// mFile << "Frame";
			// mJointDatas = mEnv->GetCharacter()->GetJointDatas();
			// auto& anglesByFrame = mJointDatas->GetAnglesByFrame();
			// std::map<std::string, std::vector<double>> avgMap;
			// for(auto iter = anglesByFrame.begin(); iter != anglesByFrame.end(); iter++)
			// {
			// 	if(iter->first == "FemurR_transverse" || iter->first == "FemurR_sagittal" || iter->first == "FemurR_frontal" || iter->first == "TibiaR_transverse" || iter->first == "TibiaR_sagittal" || iter->first == "TibiaR_frontal" || iter->first == "TalusR_transverse" || iter->first == "TalusR_sagittal" || iter->first == "TalusR_frontal")
			// 	{
			// 		mFile << " " + iter->first;
			// 		std::vector<double> avg;
			// 		for(int i=0; i<(iter->second).size(); i++)
			// 		{
			// 			double sum = 0.0;
			// 			for(int j=0; j<(iter->second).at(i).size(); j++)
			// 				sum += (iter->second).at(i).at(j);
			// 			avg.push_back(sum/(double)(iter->second).at(i).size());
			// 		}

			// 		avgMap[iter->first] = avg;
			// 	}
			// }

			// mFile << " TorqueFrame torqueNorm";
			// mFile << " metaE";
			// mMetabolicEnergy = mEnv->GetCharacter()->GetMetabolicEnergy();
			// std::vector<double> avgMap2;
			// auto& HOUD06_ByFrame = mMetabolicEnergy->GetHOUD06_ByFrame();
			// for(int i=0; i<HOUD06_ByFrame.size(); i++)
			// {
			// 	double sum = 0.0;
			// 	for(int j=0; j<HOUD06_ByFrame.at(i).size(); j++)
			// 	{
			// 		sum += HOUD06_ByFrame.at(i).at(j);
			// 	}
			// 	avgMap2.push_back(sum/(double)HOUD06_ByFrame.at(i).size());
			// }

			// auto& HOUD06_mapByFrame = mMetabolicEnergy->GetHOUD06_mapByFrame();
			// std::map<std::string, std::vector<double>> avgMap3;
			// for(auto iter = HOUD06_mapByFrame.begin(); iter != HOUD06_mapByFrame.end(); iter++)
			// {
			// 	mFile << " " + iter->first;
			// 	std::vector<double> avg;
			// 	for(int i=0; i<(iter->second).size(); i++)
			// 	{
			// 		double sum = 0.0;
			// 		for(int j=0; j<(iter->second).at(i).size(); j++)
			// 			sum += (iter->second).at(i).at(j);
			// 		avg.push_back(sum/(double)(iter->second).at(i).size());
			// 	}
			// 	avgMap3[iter->first] = avg;
			// }

			// mJointDatas = mEnv->GetCharacter()->GetJointDatas();
			// std::deque<double> torquesNormCycle = mJointDatas->GetTorquesNormCycle();
			// std::deque<int> torquesFrame = mJointDatas->GetTorquesFrame();

			// for(int i=0; i<34; i++)
			// {
			// 	mFile << "\n";
			// 	mFile << std::to_string(i);
			// 	for(auto iter = anglesByFrame.begin(); iter != anglesByFrame.end(); iter++)
			// 	{
			// 		if(iter->first == "FemurR_transverse" || iter->first == "FemurR_sagittal" || iter->first == "FemurR_frontal" || iter->first == "TibiaR_transverse" || iter->first == "TibiaR_sagittal" || iter->first == "TibiaR_frontal" || iter->first == "TalusR_transverse" || iter->first == "TalusR_sagittal" || iter->first == "TalusR_frontal")
			// 		{
			// 			mFile << " " + std::to_string(avgMap[iter->first].at(i)*180.0/M_PI);
			// 		}
			// 	}

			// 	mFile << " " + std::to_string(torquesFrame.at(i));
			// 	mFile << " " + std::to_string(torquesNormCycle.at(i));

			// 	mFile << " " + std::to_string(avgMap2.at(i));

			// 	for(auto iter = HOUD06_mapByFrame.begin(); iter != HOUD06_mapByFrame.end(); iter++)
			// 	{
			// 		mFile << " "+ std::to_string(avgMap3[iter->first].at(i));
			// 	}
			// }
			// mFile.close();
			// std::cout << mFileName << " close" << std::endl;

			isOpenFile = false;
		}
	}
}



// void
// Window::
// Write()
// {
// 	if(mWriteFile)
// 	{
// 		if(!isOpenFile)
// 		{
// 			this->WriteFileName();

// 			mFile.open(mFileName);
// 			if(mFile.is_open())
// 				std::cout << mFileName << " open" << std::endl;

// 			mFile << "Time Frame AdtFrame";
// 			this->WriteMetaE();
// 			this->WriteJointAngle();
// 			// this->WriteJointTorque();

// 			isOpenFile = true;
// 		}

// 		if(isOpenFile)
// 		{
// 			double t = mEnv->GetWorld()->getTime();
// 			double f = mEnv->GetCharacter()->GetFrame();
// 			double af = mEnv->GetCharacter()->GetAdaptiveFrame();
// 			mFile << "\n";
// 			mFile << std::to_string(t) + " " + std::to_string(f) + " " + std::to_string(af);

// 			this->WriteMetaE();
// 			this->WriteJointAngle();
// 		}
// 	}
// 	else
// 	{
// 		if(isOpenFile)
// 		{
// 			mFile.close();
// 			std::cout << mFileName << " close" << std::endl;

// 			mFileName += "_avg";
// 			mFile.open(mFileName);
// 			if(mFile.is_open())
// 				std::cout << mFileName << " open" << std::endl;

// 			mFile << "Frame";
// 			mJointDatas = mEnv->GetCharacter()->GetJointDatas();
// 			auto& anglesByFrame = mJointDatas->GetAnglesByFrame();
// 			std::map<std::string, std::vector<double>> avgMap;
// 			for(auto iter = anglesByFrame.begin(); iter != anglesByFrame.end(); iter++)
// 			{
// 				if(iter->first == "FemurR_transverse" || iter->first == "FemurR_sagittal" || iter->first == "FemurR_frontal" || iter->first == "TibiaR_transverse" || iter->first == "TibiaR_sagittal" || iter->first == "TibiaR_frontal" || iter->first == "TalusR_transverse" || iter->first == "TalusR_sagittal" || iter->first == "TalusR_frontal")
// 				{
// 					mFile << " " + iter->first;
// 					std::vector<double> avg;
// 					for(int i=0; i<(iter->second).size(); i++)
// 					{
// 						double sum = 0.0;
// 						for(int j=0; j<(iter->second).at(i).size(); j++)
// 							sum += (iter->second).at(i).at(j);
// 						avg.push_back(sum/(double)(iter->second).at(i).size());
// 					}

// 					avgMap[iter->first] = avg;
// 				}
// 			}

// 			mFile << " TorqueFrame torqueNorm";
// 			mFile << " metaE";
// 			mMetabolicEnergy = mEnv->GetCharacter()->GetMetabolicEnergy();
// 			std::vector<double> avgMap2;
// 			auto& HOUD06_ByFrame = mMetabolicEnergy->GetHOUD06_ByFrame();
// 			for(int i=0; i<HOUD06_ByFrame.size(); i++)
// 			{
// 				double sum = 0.0;
// 				for(int j=0; j<HOUD06_ByFrame.at(i).size(); j++)
// 				{
// 					sum += HOUD06_ByFrame.at(i).at(j);
// 				}
// 				avgMap2.push_back(sum/(double)HOUD06_ByFrame.at(i).size());
// 			}

// 			auto& HOUD06_mapByFrame = mMetabolicEnergy->GetHOUD06_mapByFrame();
// 			std::map<std::string, std::vector<double>> avgMap3;
// 			for(auto iter = HOUD06_mapByFrame.begin(); iter != HOUD06_mapByFrame.end(); iter++)
// 			{
// 				mFile << " " + iter->first;
// 				std::vector<double> avg;
// 				for(int i=0; i<(iter->second).size(); i++)
// 				{
// 					double sum = 0.0;
// 					for(int j=0; j<(iter->second).at(i).size(); j++)
// 						sum += (iter->second).at(i).at(j);
// 					avg.push_back(sum/(double)(iter->second).at(i).size());
// 				}
// 				avgMap3[iter->first] = avg;
// 			}

// 			mJointDatas = mEnv->GetCharacter()->GetJointDatas();
// 			std::deque<double> torquesNormCycle = mJointDatas->GetTorquesNormCycle();
// 			std::deque<int> torquesFrame = mJointDatas->GetTorquesFrame();

// 			for(int i=0; i<34; i++)
// 			{
// 				mFile << "\n";
// 				mFile << std::to_string(i);
// 				for(auto iter = anglesByFrame.begin(); iter != anglesByFrame.end(); iter++)
// 				{
// 					if(iter->first == "FemurR_transverse" || iter->first == "FemurR_sagittal" || iter->first == "FemurR_frontal" || iter->first == "TibiaR_transverse" || iter->first == "TibiaR_sagittal" || iter->first == "TibiaR_frontal" || iter->first == "TalusR_transverse" || iter->first == "TalusR_sagittal" || iter->first == "TalusR_frontal")
// 					{
// 						mFile << " " + std::to_string(avgMap[iter->first].at(i)*180.0/M_PI);
// 					}
// 				}

// 				mFile << " " + std::to_string(torquesFrame.at(i));
// 				mFile << " " + std::to_string(torquesNormCycle.at(i));

// 				mFile << " " + std::to_string(avgMap2.at(i));

// 				for(auto iter = HOUD06_mapByFrame.begin(); iter != HOUD06_mapByFrame.end(); iter++)
// 				{
// 					mFile << " "+ std::to_string(avgMap3[iter->first].at(i));
// 				}
// 			}
// 			mFile.close();
// 			std::cout << mFileName << " close" << std::endl;

// 			isOpenFile = false;
// 		}
// 	}
// }

void
Window::
GraphMode()
{
	mMetabolicEnergyMode = (mMetabolicEnergyMode+1)%48;
	mJointTorqueMode = (mJointTorqueMode+1)%9;
	mJointAngleMode = (mJointAngleMode+1)%11;
}

void
Window::
keyboard(unsigned char _key, int _x, int _y)
{
	double f_ = 0.0;
	switch (_key)
	{
	case 'r': this->Reset();break;
	case 's': this->Step();break;
	case 'f': mFocus = !mFocus;break;
	case 'o': mDrawOBJ = !mDrawOBJ;break;
	case 'g': mDrawGraph = !mDrawGraph;break;
	case 'a': mDrawArrow = !mDrawArrow;break;
	case 't': mDrawTarget = !mDrawTarget;break;
	case 'T': mDrawReference = !mDrawReference;break;
	case ' ': mSimulating = !mSimulating;break;
	case 'c': mDrawCharacter = !mDrawCharacter;break;
	case 'p': mDrawParameter = !mDrawParameter;break;
	case 'd':
		if(mEnv->GetUseDevice())
			mDevice_On = !mDevice_On;
		break;
	case 'v' : mViewMode = (mViewMode+1)%4;break;
	case '\t': this->GraphMode(); break;
	case 'm' : this->m(true); break;
	case 'n' : this->m(false); break;
	case 'x' : mWriteFile = true; break;
	case 'z' : mWriteFile = false; break;
	case '`' : mCharacterMode = (mCharacterMode+1)%2; break;
	case '1' : mParamMode = 1; break;
	case '2' : mParamMode = 2; break;
	case '3' : mParamMode = 3; break;
	case '4' : mParamMode = 4; break;
	case '5' : mParamMode = 5; break;
	case '+' : ParamChange(true); break;
	case '-' : ParamChange(false); break;
	case 27 : exit(0);break;
	default:
		Win3D::keyboard(_key,_x,_y);break;
	}
}

void
Window::
displayTimer(int _val)
{
	if(mSimulating){
		Step();
	}

	glutPostRedisplay();
	glutTimerFunc(mDisplayTimeout, refreshTimer, _val);
}

void
Window::
Reset()
{
	mFootprint.clear();
	mFootinterval.clear();
	mFootinterval.resize(20);
	mEnv->Reset();
	mDisplayIter = 0;
}

void
Window::
Step()
{
	int num = mEnv->GetNumSteps()/2.0;
	Eigen::VectorXd action;
	Eigen::VectorXd action_device;

	if(mDisplayIter % 2 == 0)
	{
		if(mNNLoaded)
			action = GetActionFromNN();
		else
			action = Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetNumAction());

		if(mDeviceNNLoaded)
			action_device = GetActionFromNN_Device();

		mEnv->SetAction(action);
		if(mDeviceNNLoaded)
			mEnv->GetDevice()->SetAction(action_device);

		// this->Write();
		mDisplayIter = 0;
	}

	if(mEnv->GetUseMuscle())
	{
		int inference_per_sim = 2;
		for(int i=0; i<num; i+=inference_per_sim){
			Eigen::VectorXd mt = mEnv->GetCharacter()->GetMuscleTorques();
			mEnv->GetCharacter()->SetActivationLevels(GetActivationFromNN(mt));
			for(int j=0; j<inference_per_sim; j++)
				mEnv->Step(mDevice_On, true);
		}
	}
	else
	{
		for(int i=0; i<num; i++){
			this->Write();
			mEnv->Step(mDevice_On, true);
		}
	}

	mEnv->GetReward();
	// this->SetTrajectory();
	mDisplayIter++;
	// glutPostRedisplay();
}

void
Window::
SetTrajectory()
{
	const SkeletonPtr& skel = mEnv->GetCharacter()->GetSkeleton();
	const BodyNode* talusR = skel->getBodyNode("TalusR");
	const BodyNode* talusL = skel->getBodyNode("TalusL");

	if(talusR->getCOM()[1] > 0.035)
		mTalusR = false;
	if(talusR->getCOM()[1] < 0.035 && !mTalusR)
	{
		mFootprint.push_back(talusR->getCOM());
		int idx = mFootprint.size()-1;
		if(idx > 0)
		{
			Eigen::Vector3d prev = mFootprint[idx-1];
			Eigen::Vector3d cur = mFootprint[idx];
			double len = (cur-prev).norm();
			mFootinterval.pop_back();
			mFootinterval.push_front(len);
		}
		mTalusR = true;
	}

	if(talusL->getCOM()[1] > 0.035)
		mTalusL = false;
	if(talusL->getCOM()[1] < 0.035 && !mTalusL)
	{
		mFootprint.push_back(talusL->getCOM());
		int idx = mFootprint.size()-1;
		if(idx > 0)
		{
			Eigen::Vector3d prev = mFootprint[idx-1];
			Eigen::Vector3d cur = mFootprint[idx];
			double len = (cur-prev).norm();
			mFootinterval.pop_back();
			mFootinterval.push_front(len);
		}
		mTalusL = true;
	}
}

void
Window::
SetFocus()
{
	if(mFocus)
	{
		mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();
		mTrans[1] = -0.931252;
		mTrans *= 1000.0;
		Eigen::Quaterniond origin_r = mTrackBall.getCurrQuat();
		if (mViewMode == 0)
		{
			mTrackBall.setQuaternion(Eigen::Quaterniond::Identity());
			Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(1 * 0.05 * M_PI, Eigen::Vector3d::UnitX()));
			mTrackBall.setQuaternion(r);
		}
		else if (mViewMode == 2 && Eigen::AngleAxisd(origin_r).angle() < 0.5 * M_PI)
		{
			Eigen::Vector3d axis(0.0, cos(0.05*M_PI), sin(0.05*M_PI));
			Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(1 * 0.01 * M_PI, axis)) * origin_r;
			mTrackBall.setQuaternion(r);
			}
		else if (mViewMode == 3 && Eigen::AngleAxisd(origin_r).axis()[1] > 0)
		{
			Eigen::Vector3d axis(0.0, cos(0.05*M_PI), sin(0.05*M_PI));
			Eigen::Quaterniond r = Eigen::Quaterniond(Eigen::AngleAxisd(-1 * 0.01 * M_PI, axis)) * origin_r;
			mTrackBall.setQuaternion(r);
		}
	}
}

void
Window::
SetViewMatrix()
{
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);

	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<
	matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<
	matrix[12],matrix[13],matrix[14];

	mViewMatrix.linear() = A;
	mViewMatrix.translation() = b;
}

void
Window::
draw()
{
	SetViewMatrix();

	DrawGround();
	DrawCharacter();
	if(mEnv->GetUseDevice())
		DrawDevice();

	if(mEnv->GetNumParamState() > 0 && mDrawParameter)
		DrawParameter();

	// DrawTrajectory();
	// DrawStride();
	SetFocus();
}

void
Window::
DrawContactForce()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);

	// Eigen::Vector4d color(0.8, 1.2, 0.8, 0.6);
	// mRI->setPenColor(color);

	// Contact* contact = mEnv->GetCharacter()->GetContacts();
	// auto& objects = contact->GetContactObjects();
	// double force_scaler = -0.0005;
	// for(auto iter = objects.begin(); iter != objects.end(); iter++)
	// {
	// 	for(int i=0; i<(iter->second).size(); i++)
	// 	{
	// 		Eigen::Vector3d f = force_scaler*((iter->second).at(i)).first;
	// 		Eigen::Vector3d p = ((iter->second).at(i)).second;
	// 		double norm = f.norm();
	// 		// if(norm != 0)
	// 		// 	dart::gui::drawArrow3D(p, f/norm, norm, 0.005, 0.015);
	// 	}
	// }

	// glDisable(GL_COLOR_MATERIAL);
	// glDisable(GL_DEPTH_TEST);

	// DrawGLBegin();

	// double fL = contact->GetContactForce("TalusL");
	// double fR = contact->GetContactForce("TalusR");

	// bool big = true;
	// DrawString(0.70, 0.44, big, "Contact L : " + std::to_string(fL));
	// DrawString(0.70, 0.41, big, "Contact R : " + std::to_string(fR));

	DrawGLEnd();
}

void
Window::
DrawCoT()
{
	DrawGLBegin();

	double cot = mEnv->GetCharacter()->GetCoT();
	bool big = true;

	DrawString(0.70, 0.47, big, "CoT : " + std::to_string(cot));

	DrawGLEnd();
}

void
Window::
DrawVelocity()
{
	DrawGLBegin();

	double vel = mEnv->GetCharacter()->GetCurVelocity();
	double vel_h = vel;
	bool big = true;

	DrawString(0.70, 0.50, big, "Velocity : " + std::to_string(vel_h) + " m/s");

	DrawGLEnd();
}

void
Window::
DrawStride()
{
	DrawGLBegin();

	double strideL = mEnv->GetCharacter()->GetStrideL();
	double strideR = mEnv->GetCharacter()->GetStrideR();
	bool big = true;

	DrawString(0.70, 0.47, big, "Stride L : " + std::to_string(strideL) + " m");
	DrawString(0.70, 0.44, big, "Stride R : " + std::to_string(strideR) + " m");

	DrawGLEnd();
}

void
Window::
DrawTime()
{
	DrawGLBegin();

	double t = mEnv->GetWorld()->getTime();
	t -= 1.0/(double)(2.0*mEnv->GetControlHz());
	if(t<0)
		t = 0.0;
	double f = mEnv->GetCharacter()->GetFrame();
	double p = mEnv->GetCharacter()->GetPhase();
	double af = mEnv->GetCharacter()->GetAdaptiveFrame();
	double ap = mEnv->GetCharacter()->GetAdaptivePhase();
	bool big = true;
	DrawString(0.47, 0.93, big, "Time : " + std::to_string(t) + " s");
	DrawString(0.44, 0.90, big, "Ref Phase : " + std::to_string(p));
	DrawString(0.44, 0.87, big, "Adt Phase : " + std::to_string(ap));

	DrawGLEnd();
}

void
Window::
DrawParameter()
{
	DrawGLBegin();

	double w = 0.20;
	double h = 0.34;
	double h_offset = 0.20;
	double x = 0.69;
	double y = 0.62;

	DrawQuads(x, y, w, h, white);

	if(mEnv->GetCharacter()->GetNumParamState() > 0)
	{
		Eigen::VectorXd min_v = mEnv->GetCharacter()->GetMinV();
		Eigen::VectorXd max_v = mEnv->GetCharacter()->GetMaxV();

		double m_ratio = mEnv->GetCharacter()->GetMassRatio();
		DrawQuads(x+0.01, y+0.01, 0.02, (m_ratio)*h_offset, red);
		DrawQuads(x+0.01, y+0.01+(m_ratio)*h_offset, 0.02, (max_v[0]-m_ratio)*h_offset, red_trans);

		double f_ratio = mEnv->GetCharacter()->GetForceRatio();
		DrawQuads(x+0.05, y+0.01, 0.02, (f_ratio)*h_offset, yellow);
		DrawQuads(x+0.05, y+0.01+(f_ratio)*h_offset, 0.02, (max_v[1]-f_ratio)*h_offset, yellow_trans);

		double s_ratio = mEnv->GetCharacter()->GetSpeedRatio();
		DrawQuads(x+0.09, y+0.01, 0.02, (s_ratio)*h_offset, green);
		DrawQuads(x+0.09, y+0.01+(s_ratio)*h_offset, 0.02, (max_v[2]-s_ratio)*h_offset, green_trans);
		// if(max_v[2] != min_v[2])
		// 	max_v[2] -= 0.0999;
		// DrawQuads(x+0.09, y+0.01+(s_ratio)*h_offset, 0.02, (max_v[2]-s_ratio)*h_offset, green_trans);

		DrawString(x+0.00, y+(m_ratio)*h_offset+0.02, std::to_string(m_ratio));
		DrawString(x+0.00, y-0.02, "Mass");

		DrawString(x+0.05, y+(f_ratio)*h_offset+0.02, std::to_string(f_ratio));
		DrawString(x+0.04, y-0.02, "Force");

		if(s_ratio > max_v[2])
			s_ratio = max_v[2];
		DrawString(x+0.09, y+(s_ratio)*h_offset+0.02, std::to_string(s_ratio));
		DrawString(x+0.078, y-0.02, "Speed");
	}

	if(mEnv->GetUseDevice() && mEnv->GetDevice()->GetNumParamState() > 0)
	{
		Eigen::VectorXd min_v_dev = mEnv->GetDevice()->GetMinV();
		Eigen::VectorXd max_v_dev = mEnv->GetDevice()->GetMaxV();

		double k_ = mEnv->GetDevice()->GetK_();
		DrawQuads(x+0.13, y+0.01, 0.02, (k_/30.0)*h_offset, blue);
		DrawQuads(x+0.13, y+0.01+(k_/30.0)*h_offset, 0.02, (max_v_dev[0]-k_/30.0)*h_offset, blue_trans);

		double t_ = mEnv->GetDevice()->GetDelta_t();
		DrawQuads(x+0.17, y+0.01, 0.02, 3.333*t_*h_offset, purple);
		DrawQuads(x+0.17, y+0.01+3.333*t_*h_offset, 0.02, 3.333*(max_v_dev[1]-t_)*h_offset, purple_trans);

		DrawString(x+0.13, y+(k_/30.0)*h_offset+0.02, std::to_string(k_));
		DrawString(x+0.12, y-0.02, "Device");

		DrawString(x+0.17, y+3.333*t_*h_offset+0.02, std::to_string(t_));
		DrawString(x+0.165, y-0.02, "Delta t");
	}

	DrawGLEnd();
}

void
Window::
DrawGround()
{
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glDisable(GL_LIGHTING);

	auto ground = mEnv->GetGround();
	float y = ground->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(ground->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;

	int count = 0;
	double w = 2.0;
	double h = 2.0;
	for(double x=-100.0; x<=100.0; x+=2.0)
	{
		for(double z=-100.0; z<=100.0; z+=2.0)
		{
			if(count%2==0)
				glColor3f(0.82, 0.80, 0.78);
			else
				glColor3f(0.72, 0.70, 0.68);

			glBegin(GL_QUADS);
			glVertex3f(x  , y, z  );
			glVertex3f(x+w, y, z  );
			glVertex3f(x+w, y, z+h);
			glVertex3f(x  , y, z+h);
			glEnd();

			count++;
		}
	}
	glEnable(GL_LIGHTING);
}

void
Window::
DrawCharacter()
{
	SkeletonPtr skeleton = mEnv->GetCharacter()->GetSkeleton();

	if(mDrawCharacter)
	{
		isDrawCharacter = true;
		DrawSkeleton(skeleton);
		if(mEnv->GetUseMuscle())
			DrawMuscles(mEnv->GetCharacter()->GetMuscles());
		isDrawCharacter = false;
	}

	if(mDrawTarget)
		DrawTarget();

	if(mDrawReference)
		DrawReference();

	if(mDrawGraph){
		// if(mEnv->GetUseMuscle())
		// 	this->DrawMetabolicEnergy_();
		// else if(mEnv->GetUseDevice())
		// 	this->DrawFemurSignals();

		// this->DrawJointAngles();
		this->DrawReward();

		if(mEnv->GetUseMuscle())
			this->DrawMetabolicEnergys();
		// else
		// this->DrawJointTorques();
	}
	else{
		this->DrawVelocity();
		this->DrawStride();

		// this->DrawCoT();
		// this->DrawContactForce();
		// this->DrawMetabolicEnergy();
	}
	this->DrawTime();
}

void
Window::
DrawSkeleton(const SkeletonPtr& skel)
{
	DrawBodyNode(skel->getRootBodyNode());
}

void
Window::
DrawBodyNode(const BodyNode* bn)
{
	if(!bn)
		return;
	if(!mRI)
		return;

	mRI->pushMatrix();
	mRI->transform(bn->getRelativeTransform());

	auto sns = bn->getShapeNodesWith<VisualAspect>();
	if(bn->getName() == "FemurL" || bn->getName() == "FemurR" || bn->getName() == "Pelvis")
		mDrawCoordinate = true;
	else
		mDrawCoordinate = false;

	for(const auto& sn : sns)
		DrawShapeFrame(sn);

	for(const auto& et : bn->getChildEntities())
		DrawEntity(et);

	mRI->popMatrix();
}

void
Window::
DrawShapeFrame(const ShapeFrame* sf)
{
	if(!sf)
		return;

	if(!mRI)
		return;

	const auto& va = sf->getVisualAspect();

	if(!va || va->isHidden())
		return;

	mRI->pushMatrix();
	mRI->transform(sf->getRelativeTransform());

	mColor = va->getRGBA();
	if(isDrawCharacter)
	{
		if(mDrawOBJ)
			mColor << 0.75, 0.75, 0.75, 0.3;
		else
			mColor[3] = 0.3;
	} 
	if(isDrawTarget)
		mColor << 1.0, 0.6, 0.6, 0.3;
	if(isDrawReference)
		mColor << 0.6, 1.0, 0.6, 0.3;
	if(isDrawDevice)
		mColor << 0.3, 0.3, 0.3, 1.0;
	DrawShape(sf->getShape().get(), mColor);
	mRI->popMatrix();
}

void
Window::
DrawShape(const Shape* shape, const Eigen::Vector4d& color)
{
	if(!shape)
		return;
	if(!mRI)
		return;

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	mRI->setPenColor(color);
	if(mDrawOBJ == false)
	{
		if (shape->is<SphereShape>())
		{
			const auto* sphere = static_cast<const SphereShape*>(shape);
			// mRI->drawSphere(sphere->getRadius());
		}
		else if (shape->is<BoxShape>())
		{
			const auto* box = static_cast<const BoxShape*>(shape);
			mRI->drawCube(box->getSize());
		}
		else if (shape->is<CapsuleShape>())
		{
			const auto* capsule = static_cast<const CapsuleShape*>(shape);
			mRI->drawCapsule(capsule->getRadius(), capsule->getHeight());
		}
	}
	else
	{
		if (shape->is<MeshShape>())
		{
			const auto& mesh = static_cast<const MeshShape*>(shape);
			float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
			mShapeRenderer.renderMesh(mesh, false, y, color);			
		}
	}

	if(mDrawCoordinate && shape->is<SphereShape>()){
		DrawLine3D(0.0,0.0,0.0, 0.1, 0.0, 0.0, red, 3.0);
		DrawLine3D(0.0,0.0,0.0, 0.0, 0.1, 0.0, green, 3.0);
		DrawLine3D(0.0,0.0,0.0, 0.0, 0.0, 0.1, blue, 3.0);
	}

	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_DEPTH_TEST);
}

void
Window::
DrawEntity(const Entity* entity)
{
	if (!entity)
		return;
	const auto& bn = dynamic_cast<const BodyNode*>(entity);
	if(bn)
	{
		DrawBodyNode(bn);
		return;
	}

	const auto& sf = dynamic_cast<const ShapeFrame*>(entity);
	if(sf)
	{
		DrawShapeFrame(sf);
		return;
	}
}

void
Window::
DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y)
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glScalef(scale[0],scale[1],scale[2]);
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];

	Eigen::Affine3d M;
	M.linear() = A;
	M.translation() = b;
	M = (mViewMatrix.inverse()) * M;

	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(mViewMatrix.data());
	DrawAiMesh(mesh,mesh->mRootNode,M,y);
	glPopMatrix();
	glPopMatrix();
	glEnable(GL_LIGHTING);
}

void
Window::
DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y)
{
	unsigned int i;
	unsigned int n = 0, t;
	Eigen::Vector3d v;
	Eigen::Vector3d dir(0.4,0,-0.4);
	glColor3f(0.3,0.3,0.3);

	// update transform
	// draw all meshes assigned to this node
	for (; n < nd->mNumMeshes; ++n) {
		const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

		for (t = 0; t < mesh->mNumFaces; ++t) {
			const struct aiFace* face = &mesh->mFaces[t];
			GLenum face_mode;

			switch(face->mNumIndices) {
				case 1: face_mode = GL_POINTS; break;
				case 2: face_mode = GL_LINES; break;
				case 3: face_mode = GL_TRIANGLES; break;
				default: face_mode = GL_POLYGON; break;
			}
			glBegin(face_mode);
			for (i = 0; i < face->mNumIndices; i++)
			{
				int index = face->mIndices[i];

				v[0] = (&mesh->mVertices[index].x)[0];
				v[1] = (&mesh->mVertices[index].x)[1];
				v[2] = (&mesh->mVertices[index].x)[2];
				v = M*v;
				double h = v[1]-y;

				v += h*dir;

				v[1] = y+0.001;
				glVertex3f(v[0],v[1],v[2]);
			}
			glEnd();
		}
	}

	for (n = 0; n < nd->mNumChildren; ++n) {
		DrawAiMesh(sc, nd->mChildren[n],M,y);
	}
}

void
Window::
DrawMuscles(const std::vector<Muscle*>& muscles)
{
	int count = 0;
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	for (auto muscle : muscles)
	{
        double a = muscle->GetActivation();
       	Eigen::Vector4d color(1.0+(3.0*a), 1.0, 1.0, 1.0);
       	glColor4dv(color.data());

        mShapeRenderer.renderMuscle(muscle);
	}
	std::map<std::string, std::vector<Muscle*>> map = mEnv->GetCharacter()->GetMusclesMap();

	double l_mt0;
	double f0;
	for (auto muscle : map[cur_muscle])
	{
	   	Eigen::Vector4d color(1.0, 4.0, 1.0, 1.0);
	   	double a = muscle->GetActivation();
	   	l_mt0 = muscle->GetMt0();
	   	f0 = muscle->GetF0();
       	// Eigen::Vector4d color(1.0+(3.0*a), 1.0, 1.0, 1.0);
       	glColor4dv(color.data());
	    mShapeRenderer.renderMuscle(muscle);
	}

	glEnable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);

	// DrawGLBegin();
	// bool big = true;
	// DrawString(0.70, 0.29, big, "Muscle : " + cur_muscle);
	// DrawString(0.70, 0.26, big, "f0 : " + std::to_string(f0));
	// DrawGLEnd();
	// DrawString(0.70, 0.29, big, "l_mt0 : " + std::to_string(l_mt0));
}

void
Window::
DrawTarget()
{
	isDrawTarget = true;

	Character* character = mEnv->GetCharacter();
	SkeletonPtr skeleton = character->GetSkeleton();

	Eigen::VectorXd cur_pos = skeleton->getPositions();

	skeleton->setPositions(character->GetTargetPositions());
	DrawBodyNode(skeleton->getRootBodyNode());

	skeleton->setPositions(cur_pos);

	isDrawTarget = false;
}

void
Window::
DrawReference()
{
	isDrawReference = true;

	Character* character = mEnv->GetCharacter();
	SkeletonPtr skeleton = character->GetSkeleton();

	Eigen::VectorXd cur_pos = skeleton->getPositions();

	skeleton->setPositions(character->GetReferencePositions());
	// skeleton->setPositions(character->GetReferenceOriginalPositions());
	DrawBodyNode(skeleton->getRootBodyNode());

	skeleton->setPositions(cur_pos);

	isDrawReference = false;
}

void
Window::
DrawTrajectory()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// Eigen::Vector4d color(0.5, 0.5, 0.5, 0.7);
	// mRI->setPenColor(color);
	// mRI->setLineWidth(8.0);

	// glBegin(GL_LINE_STRIP);
	// for(int i=0; i<mTrajectory.size(); i++)
	// {
	// 	glVertex3f(mTrajectory[i][0], 0.0, mTrajectory[i][2]);
	// }
	// glEnd();

	Eigen::Vector4d color1(0.5, 0.8, 0.5, 0.7);
	mRI->setPenColor(color1);
	for(int i=0; i<mFootprint.size(); i++)
	{
		glPushMatrix();
		glTranslatef(mFootprint[i][0], 0.0, mFootprint[i][2]);
		glutSolidCube(0.05);
		glPopMatrix();
	}

	glDisable(GL_COLOR_MATERIAL);
}

void
Window::
DrawFemurSignals()
{
	DrawGLBegin();

	double p_w  = 0.30, p_h  = 0.14;
	double pl_x = 0.69,	pl_y = 0.84;
	double pr_x = 0.69,	pr_y = 0.68;

	double offset_x = 0.00024;
	double offset_y = 0.0006;
	double offset = 0.005;
	double ratio_y = 0.3;

	Character* character = mEnv->GetCharacter();
	std::deque<double> data_L_femur = character->GetSignals(0);
	std::deque<double> data_R_femur = character->GetSignals(1);

	DrawBaseGraph(pl_x, pl_y, p_w, p_h, ratio_y, offset, "Femur L");
	DrawLineStrip(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, red, 2.0, data_L_femur);
	DrawStringMax(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, data_L_femur, red);

	DrawBaseGraph(pr_x, pr_y, p_w, p_h, ratio_y, offset, "Femur R");
	DrawLineStrip(pr_x, pr_y, p_h, ratio_y, offset_x, offset_y, offset, red, 2.0, data_R_femur);
	DrawStringMax(pr_x, pr_y, p_h, ratio_y, offset_x, offset_y, offset, data_R_femur, red);

	DrawGLEnd();
}

void
Window::
DrawJointAngles()
{
	DrawGLBegin();

	double p_w = 0.30, p_h = 0.14, p_x = 0.01, p_y = 0.84;
	double offset_y = 0.16;

	mJointDatas = mEnv->GetCharacter()->GetJointDatas();
	int modeNum = mJointDatas->GetAngles().size() / 6;
	auto iter = mJointDatas->GetAngles().begin();
	for(int i=0; i<modeNum; i++)
	{
		for(int j=0; j<6; j++)
		{
			if(mJointAngleMode == i){
				if(j%6 == 0){
					DrawAngleGraph(iter->first, iter->second, p_w, p_h, p_x, p_y-j*offset_y);
				}
				else if(j%6 == 1){
					DrawAngleGraph(iter->first, iter->second, p_w, p_h, p_x, p_y-j*offset_y);
				}
				else if(j%6 == 2){
					DrawAngleGraph(iter->first, iter->second, p_w, p_h, p_x, p_y-j*offset_y);
				}
				else if(j%6 == 3){
					DrawAngleGraph(iter->first, iter->second, p_w, p_h, p_x, p_y-j*offset_y);
				}
				else if(j%6 == 4){
					DrawAngleGraph(iter->first, iter->second, p_w, p_h, p_x, p_y-j*offset_y);
				}
				else{
					DrawAngleGraph(iter->first, iter->second, p_w, p_h, p_x, p_y-j*offset_y);
				}
			}
			iter++;
		}
	}

	DrawGLEnd();
}


void
Window::
DrawJointTorques()
{
	DrawGLBegin();

	double p_w = 0.30, p_h = 0.14, p_x = 0.01, p_y = 0.84;
	double offset_y = 0.16;

	mJointDatas = mEnv->GetCharacter()->GetJointDatas();
	int modeNum = mJointDatas->GetTorques().size() / 6;
	auto iter = mJointDatas->GetTorques().begin();
	for(int i=0; i<modeNum; i++)
	{
		for(int j=0; j<6; j++)
		{
			if(mJointTorqueMode == i)
				DrawTorqueGraph(iter->first, iter->second, p_w, p_h, p_x, p_y-j*offset_y);
			iter++;
		}
	}

	DrawGLEnd();
}

void
Window::
DrawMetabolicEnergy()
{
	DrawGLBegin();

	//Metabolic Energy Rate
	mMetabolicEnergy = mEnv->GetCharacter()->GetMetabolicEnergy();
	double BHAR04 = mMetabolicEnergy->GetBHAR04();
	double HOUD06 = mMetabolicEnergy->GetHOUD06();
	bool big = true;

	DrawString(0.70, 0.38, big, "BHAR04 : " + std::to_string(BHAR04));
	DrawString(0.70, 0.35, big, "HOUD06 : " + std::to_string(HOUD06));
	DrawGLEnd();
}

void
Window::
DrawMetabolicEnergy_()
{
	DrawGLBegin();

	double p_w  = 0.30, p_h = 0.17;
	double pl_x = 0.69, pl_y = 0.81;
	double pr_x = 0.69, pr_y = 0.62;

	double offset_x = 0.0090;
	double offset_y = 0.0010;
	double offset = 0.005;
	double ratio_y = 0.2;

	mMetabolicEnergy = mEnv->GetCharacter()->GetMetabolicEnergy();
	std::deque<double> BHAR04 = mMetabolicEnergy->GetBHAR04_deque();
	std::deque<double> HOUD06 = mMetabolicEnergy->GetHOUD06_deque();

	offset_x = (p_w - 2*offset)/(double)(HOUD06.size()-1);

	DrawBaseGraph(pl_x, pl_y, p_w, p_h, ratio_y, offset, "BHAR04");
	DrawLineStrip(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, red, 2.0, BHAR04);
	DrawStringMax(pl_x, pl_y, p_h, ratio_y, offset_x, offset_y, offset, BHAR04, red);

	DrawBaseGraph(pr_x, pr_y, p_w, p_h, ratio_y, offset, "HOUD06");
	DrawLineStrip(pr_x, pr_y, p_h, ratio_y, offset_x, offset_y, offset, red, 2.0, HOUD06);
	DrawStringMax(pr_x, pr_y, p_h, ratio_y, offset_x, offset_y, offset, HOUD06, red);

	DrawGLEnd();
}

void
Window::
DrawMetabolicEnergys()
{
	DrawGLBegin();

	double p_w = 0.30, p_h = 0.14, p_x = 0.01, p_y = 0.84;
	double offset_y = 0.16;

	mMetabolicEnergy = mEnv->GetCharacter()->GetMetabolicEnergy();
	int modeNum = mMetabolicEnergy->GetBHAR04_map_deque().size() / 6 + 1;
	auto iter_BHAR04 = mMetabolicEnergy->GetBHAR04_map_deque().begin();
	auto iter_HOUD06 = mMetabolicEnergy->GetHOUD06_map_deque().begin();
	for(int i=0; i<modeNum; i++)
	{
		for(int j=0; j<6; j++)
		{
			if(mMetabolicEnergyMode == i){
				// DrawMetabolicEnergyGraph(iter->first, iter->second, p_w, p_h, p_x, p_y-j*offset_y);
				DrawMetabolicEnergyGraph(iter_BHAR04->first, iter_BHAR04->second, iter_HOUD06->second, p_w, p_h, p_x, p_y-j*offset_y);
			}
			iter_BHAR04++;
			iter_HOUD06++;
		}
	}

	DrawGLEnd();
}

void
Window::
DrawMetabolicEnergyGraph(std::string name, std::deque<double> data, double w, double h, double x, double y)
{
	double offset_x = 0.00048;
	double offset_y = 0.0004;
	double offset = 0.005;
	double ratio_y = 0.2;

	DrawBaseGraph(x, y, w, h, ratio_y, offset, name);
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, red, 1.5, data);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, data, red);
}

void
Window::
DrawMetabolicEnergyGraph(std::string name, std::deque<double> data1, std::deque<double> data2, double w, double h, double x, double y)
{
	double offset_x = 0.0090;
	double offset_y = 0.0010;
	double offset = 0.005;
	double ratio_y = 0.2;

	DrawBaseGraph(x, y, w, h, ratio_y, offset, name);
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, red, 1.5, data1);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, data1, red);
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, blue, 1.5, data2);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, data2, blue);
}

void
Window::
DrawAngleGraph(std::string name, std::deque<double> data, double w, double h, double x, double y)
{
	int dataSize = data.size();
	double max = -180.0, min = 180.0;
	for(auto& d : data){
		d *= (180.0/M_PI);
		if(d > max)
			max = d;
		if(d < min)
			min = d;
	}

	double offset = 0.005;
	double offset_x = (w - offset)/(double)dataSize;
	double offset_y = (h - offset)/(max-min);
	double ratio_y = 0.0;

	if(min > 0){
		offset_y = (h - offset)/max;
		ratio_y = 0.0;
	}

	if(max < 0){
		offset_y = (h - offset)/(0.0 - min);
		ratio_y = 1.0;
	}

	if(max >= 0 && min <= 0)
	{
		ratio_y = -min / (max - min);
	}

	DrawBaseGraph(x, y, w, h, ratio_y, offset, name);
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, red, 1.5, data);
	// DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, data, red);
	DrawStringMinMax(x, y, h, ratio_y, offset_x, offset_y, offset, data, blue);
}


void
Window::
DrawTorqueGraph(std::string name, std::deque<double> data, double w, double h, double x, double y)
{
	double offset_x = 0.00144;
	double offset_y = 0.0005;
	double offset = 0.005;
	double ratio_y = 0.3;

	DrawBaseGraph(x, y, w, h, ratio_y, offset, name);
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, red, 1.5, data);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, data, red);
}

// void
// Window::
// DrawStride()
// {
// 	DrawGLBegin();

// 	double w = 0.15, h = 0.11, x = 0.69, y = 0.47;

// 	double offset_x = 0.003;
// 	double offset_y = 1.0;
// 	double offset = 0.005;
// 	double ratio_y = 0.0;

// 	y = 0.49;
// 	DrawBaseGraph(x, y, w, h, ratio_y, offset, "stride");
// 	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, red, 1.5, mFootinterval, 0.7, 0.8);
// 	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, mFootinterval, green, 0.7, 0.8);
// 	// DrawStringMean(x, y, w, h, ratio_y, offset_x, offset_y, offset, mFootinterval, green);

// 	DrawGLEnd();
// }

void
Window::
DrawReward()
{
	DrawGLBegin();

	double w = 0.15, h = 0.11, x = 0.69, y = 0.49;

	double offset_x = 0.002;
	double offset_y = 0.1;
	double offset = 0.005;
	double ratio_y = 0.0;

	std::map<std::string, std::deque<double>> map = mEnv->GetRewards();

	y = 0.49;
	std::deque<double> reward_c = map["reward_c"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "reward_c");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, reward_c);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, reward_c, green);

	y = 0.37;
	std::deque<double> imit_c = map["imit_c"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "imit_c");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, imit_c);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, imit_c, green);

	y = 0.25;
	std::deque<double> effi_c = map["effi_c"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "effi_c");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, effi_c);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, effi_c, green);

	y = 0.13;
	std::deque<double> effi_vel = map["effi_vel"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "effi_vel");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, effi_vel);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, effi_vel, green);

	y = 0.01;
	std::deque<double> effi_pose = map["effi_pose"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "effi_pose");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, effi_pose);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, effi_pose, green);

	x = 0.85;
	y = 0.49;
	std::deque<double> reward_s = map["reward_s"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "reward_s");
	DrawLineStrip(x, y, h, ratio_y, offset_x, 0.01*offset_y, offset, green, 1.5, reward_s);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, reward_s, green);

	// y = 0.37;
	// std::deque<double> vel = map["vel"];
	// DrawBaseGraph(x, y, w, h, ratio_y, offset, "vel");
	// DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, vel);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);

	y = 0.25;
	std::deque<double> effi_s = map["effi_s"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "effi_s");
	DrawLineStrip(x, y, h, ratio_y, offset_x, 0.01*offset_y, offset, green, 1.5, effi_s);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, effi_s, green);

	// y = 0.13;
	// std::deque<double> root = map["root"];
	// DrawBaseGraph(x, y, w, h, ratio_y, offset, "root");
	// DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, root);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);

	y = 0.01;
	std::deque<double> effi_stride = map["effi_stride"];
	DrawBaseGraph(x, y, w, h, ratio_y, offset, "effi_stride");
	DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, effi_stride);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);
	DrawStringMax(x, y, h, ratio_y, offset_x, offset_y, offset, effi_stride, green);

	// std::deque<double> contact = map["contact"];
	// DrawBaseGraph(x, y, w, h, ratio_y, offset, "contact");
	// DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, contact);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);

	// y = 0.01;
	// std::deque<double> smooth = map["smooth"];
	// DrawBaseGraph(x, y, w, h, ratio_y, offset, "smooth");
	// DrawLineStrip(x, y, h, ratio_y, offset_x, offset_y, offset, green, 1.5, smooth);
	// DrawLine(x, y+0.5*h, x+w, y+0.5*h, red_trans, 1.0);

	DrawGLEnd();
}

void
Window::
DrawDevice()
{
	Character* character = mEnv->GetCharacter();
	if(character->GetDevice_OnOff())
	{
		isDrawDevice = true;
		DrawSkeleton(mEnv->GetDevice()->GetSkeleton());
		// if(mDrawGraph)
		// 	DrawDeviceSignals();
		if(mDrawArrow)
			DrawArrow();
		isDrawDevice = false;
	}
}

void
Window::
DrawDeviceSignals()
{
	DrawGLBegin();

	double p_w = 0.30, p_h = 0.14, p_x = 0.01, p_y = 0.84;

	double offset_x = 0.00024;
	double offset_y = 0.0006;
	double offset = 0.005;
	double ratio_y = 0.3;

	Device* device = mEnv->GetDevice();
	std::deque<double> data_L = device->GetSignals(0);
	std::deque<double> data_R = device->GetSignals(1);
	if(mJointTorqueMode == 0){
		DrawLineStrip(p_x, p_y-1*0.16, p_h, ratio_y, offset_x, offset_y, offset, blue, 1.5, data_L, 180);
		DrawStringMax(p_x, p_y-1*0.16, p_h, ratio_y, offset_x, offset_y, offset, data_L, blue);
		DrawLineStrip(p_x, p_y-4*0.16, p_h, ratio_y, offset_x, offset_y, offset, blue, 1.5, data_R, 180);
		DrawStringMax(p_x, p_y-4*0.16, p_h, ratio_y, offset_x, offset_y, offset, data_R, blue);
	}

	DrawGLEnd();
}

void
Window::
DrawArrow()
{
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);

	Eigen::Vector4d color(0.6, 1.2, 0.6, 0.8);
	mRI->setPenColor(color);

	Eigen::VectorXd f = mEnv->GetDevice()->GetDesiredTorques();

	Eigen::Isometry3d trans_L = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurL")->getTransform();
	Eigen::Vector3d p_L = trans_L.translation();
	Eigen::Matrix3d rot_L = trans_L.rotation();
	Eigen::Vector3d dir_L1 = rot_L.col(2);
	Eigen::Vector3d dir_L2 = rot_L.col(2);
	dir_L2[2] *= -1;

	// if(f[6] < 0)
	// 	dart::gui::drawArrow3D(p_L, dir_L2,-0.04*f[6], 0.01, 0.03);
	// else
	// 	dart::gui::drawArrow3D(p_L, dir_L1, 0.04*f[6], 0.01, 0.03);

	Eigen::Isometry3d trans_R = mEnv->GetCharacter()->GetSkeleton()->getBodyNode("FemurR")->getTransform();
	Eigen::Vector3d p_R = trans_R.translation();
	Eigen::Matrix3d rot_R = trans_R.rotation();
	Eigen::Vector3d dir_R1 = rot_R.col(2);
	Eigen::Vector3d dir_R2 = rot_R.col(2);
	dir_R2[2] *= -1;

	// if(f[7] < 0)
	// 	dart::gui::drawArrow3D(p_R, dir_R2,-0.04*f[7], 0.015, 0.03);
	// else
	// 	dart::gui::drawArrow3D(p_R, dir_R1, 0.04*f[7], 0.015, 0.03);

	glDisable(GL_DEPTH_TEST);
	glDisable(GL_COLOR_MATERIAL);
}

void
Window::
DrawGLBegin()
{
	glGetIntegerv(GL_MATRIX_MODE, &oldMode);
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glDisable(GL_LIGHTING);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
}

void
Window::
DrawGLEnd()
{
	glDisable(GL_COLOR_MATERIAL);
	glEnable(GL_LIGHTING);
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(oldMode);
}

void
Window::
DrawBaseGraph(double x, double y, double w, double h, double ratio, double offset, std::string name)
{
	h -= 2*offset;
	w -= 2*offset;
	x += offset;
	y += offset;
	DrawQuads(x-offset, y-offset, w+2*offset, h+2*offset, white);
	DrawLine(x, y+ratio*h, x+w, y+ratio*h, black, 1.5);
	DrawLine(x, y, x, y+h, black, 1.5);
	DrawString(x+0.35*w, y+offset, name);
}

void
Window::
DrawQuads(double x, double y, double w, double h, Eigen::Vector4d color)
{
	mRI->setPenColor(color);

	glBegin(GL_QUADS);
	glVertex2f(x    , y);
	glVertex2f(x    , y + h);
	glVertex2f(x + w, y + h);
	glVertex2f(x + w, y);
	glEnd();
}

void
Window::
DrawLine(double p1_x, double p1_y, double p2_x, double p2_y, Eigen::Vector4d color, double line_width)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	glBegin(GL_LINES);
	glVertex2f(p1_x, p1_y);
	glVertex2f(p2_x, p2_y);
	glEnd();
}

void
Window::
DrawLine3D(double p1_x, double p1_y, double p1_z, double p2_x, double p2_y, double p2_z, Eigen::Vector4d color, double line_width)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	glBegin(GL_LINES);
	glVertex3f(p1_x, p1_y, p1_z);
	glVertex3f(p2_x, p2_y, p2_z);
	glEnd();
}

void
Window::
DrawLineStipple(double p1_x, double p1_y, double p2_x, double p2_y, Eigen::Vector4d color, double line_width)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	glLineStipple(3, 0xAAAA);
	glEnable(GL_LINE_STIPPLE);
	glBegin(GL_LINES);
	glVertex2f(p1_x, p1_y);
	glVertex2f(p2_x, p2_y);
	glEnd();
}

void
Window::
DrawString(double x, double y, std::string str)
{
	Eigen::Vector4d black(0.0, 0.0, 0.0, 1.0);
	mRI->setPenColor(black);

	glRasterPos2f(x, y);
	unsigned int length = str.length();
	for (unsigned int c = 0; c < length; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str.at(c));
}

void
Window::
DrawString(double x, double y, bool big, std::string str)
{
	Eigen::Vector4d black(0.0, 0.0, 0.0, 1.0);
	mRI->setPenColor(black);

	glRasterPos2f(x, y);
	unsigned int length = str.length();
	for (unsigned int c = 0; c < length; c++)
		if(big)
			glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, str.at(c));
		else
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str.at(c));
}

void
Window::
DrawString(double x, double y, std::string str, Eigen::Vector4d color)
{
	mRI->setPenColor(color);

	glRasterPos2f(x, y);
	unsigned int length = str.length();
	for (unsigned int c = 0; c < length; c++)
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, str.at(c));
}

void
Window::
DrawStringMean(double x, double y, double w, double h, double ratio, double offset_x, double offset_y, double offset, std::vector<double> data, Eigen::Vector4d color)
{
	double sum = 0;
	for(int i=0; i<data.size(); i++)
		sum += data[i];

	double mean = sum/(double)(data.size());

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+w, y+mean*offset_y, std::to_string(mean), color);

	mRI->setPenColor(color);
	mRI->setLineWidth(1.5);

	glBegin(GL_LINE_STRIP);
	glVertex2f(x,         y+mean*offset_y);
	glVertex2f(x+w-2*offset,y+mean*offset_y);
	glEnd();
}

void
Window::
DrawStringMean(double x, double y, double w, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color)
{
	double sum = 0;
	for(int i=0; i<data.size(); i++)
		sum += data.at(i);

	double mean = sum/(double)(data.size());

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+w, y+mean*offset_y, std::to_string(mean), color);

	mRI->setPenColor(color);
	mRI->setLineWidth(1.5);

	glBegin(GL_LINE_STRIP);
	glVertex2f(x,         y+mean*offset_y);
	glVertex2f(x+w-2*offset,y+mean*offset_y);
	glEnd();
}

void
Window::
DrawStringMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::vector<double> data, Eigen::Vector4d color)
{
	double max = 0;
	int idx = 0;
	for(int i=0; i<data.size(); i++)
	{
		if(data[i] > max)
		{
			max = data[i];
			idx = i;
		}
	}

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+idx*offset_x, y+max*offset_y, std::to_string(max), color);
}

void
Window::
DrawStringMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color)
{
	double max = 0;
	int idx = 0;
	for(int i=0; i<data.size(); i++)
	{
		if(data[i] > max)
		{
			max = data.at(i);
			idx = i;
		}
	}

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+idx*offset_x, y+max*offset_y, std::to_string(max), color);
}

void
Window::
DrawStringMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color, double lower, double upper)
{
	double max = 0;
	int idx = 0;
	for(int i=0; i<data.size(); i++)
	{
		if(data[i] > upper)
			data[i] = upper;

		if(data[i] < lower)
			data[i] = lower;

		if(data[i] > max)
		{
			max = data.at(i);
			idx = i;
		}
	}

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+idx*offset_x, y+(max-lower)*offset_y, std::to_string(max), color);
}

void
Window::
DrawStringMinMax(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, std::deque<double> data, Eigen::Vector4d color)
{
	double max = -10000, min = 10000;
	int idx_max = 0, idx_min = 0;
	for(int i=0; i<data.size(); i++)
	{
		if(data[i] > max)
		{
			max = data.at(i);
			idx_max = i;
		}

		if(data[i] < min)
		{
			min = data.at(i);
			idx_min = i;
		}
	}

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	DrawString(x+idx_max*offset_x, y+max*offset_y, std::to_string(max), color);
	DrawString(x+idx_min*offset_x, y+min*offset_y, std::to_string(min), color);
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, int idx)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=idx; i<data.size(); i++)
		glVertex2f(x + offset_x*(i-idx), y + offset_y*data.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::vector<double>& data, Eigen::Vector4d color1, double line_width1, std::vector<double>& data1)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data.at(i));
	glEnd();

	mRI->setPenColor(color1);
	mRI->setLineWidth(line_width1);

	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data1.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data1.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::vector<double>& data)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data.at(i));
	glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, int idx, Eigen::Vector4d color1, double line_width1, std::deque<double>& data1, int idx1)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=idx; i<data.size(); i++)
		glVertex2f(x + offset_x*(i-idx), y + offset_y*data.at(i));
	glEnd();

	mRI->setPenColor(color1);
	mRI->setLineWidth(line_width1);

	glBegin(GL_LINE_STRIP);
	for(int i=idx1; i<data1.size(); i++)
		glVertex2f(x + offset_x*(i-idx1), y + offset_y*data1.at(i));
	glEnd();

	Eigen::Vector4d blend((color[0]+color1[0])/2.0, (color[1]+color1[1])/2.0, (color[2]+color1[2])/2.0, (color[3]+color1[3])/2.0);

	// mRI->setPenColor(blend);

	// glBegin(GL_LINE_STRIP);
	// for(int i=0; i<data1.size(); i++)
	//  glVertex2f(x + offset_x*i, y + offset_y*(data.at(i)+data1.at(i)));
	// for(int i=180; i<data.size(); i++)
	//  glVertex2f(x + offset_x*(i-180), y + offset_y*(data.at(i)+0.5*data1.at(i-180)));
	// glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, Eigen::Vector4d color1, double line_width1, std::deque<double>& data1)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=180; i<data.size(); i++)
		glVertex2f(x + offset_x*(i-180), y + offset_y*data.at(i));
	glEnd();

	mRI->setPenColor(color1);
	mRI->setLineWidth(line_width1);

	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data1.size(); i++)
		glVertex2f(x + offset_x*i, y + offset_y*data1.at(i));
	glEnd();

	Eigen::Vector4d blend((color[0]+color1[0])/2.0, (color[1]+color1[1])/2.0, (color[2]+color1[2])/2.0, (color[3]+color1[3])/2.0);

	// mRI->setPenColor(blend);

	// glBegin(GL_LINE_STRIP);
	// for(int i=0; i<data1.size(); i++)
	//  glVertex2f(x + offset_x*i, y + offset_y*(data.at(i)+data1.at(i)));
	// for(int i=180; i<data.size(); i++)
	//  glVertex2f(x + offset_x*(i-180), y + offset_y*(data.at(i)+0.5*data1.at(i-180)));
	// glEnd();
}

void
Window::
DrawLineStrip(double x, double y, double h, double ratio, double offset_x, double offset_y, double offset, Eigen::Vector4d color, double line_width, std::deque<double>& data, double lower, double upper)
{
	mRI->setPenColor(color);
	mRI->setLineWidth(line_width);

	h -= 2*offset;
	x += offset;
	y += offset + ratio*h;
	glBegin(GL_LINE_STRIP);
	for(int i=0; i<data.size(); i++)
	{
		if(data.at(i) < lower)
			glVertex2f(x + offset_x*i, y);
		else if(data.at(i) > upper)
			glVertex2f(x + offset_x*i, y + offset_y*(upper));
		else
			glVertex2f(x + offset_x*i, y + offset_y*(data.at(i)-lower));
	}
	glEnd();
}


Eigen::VectorXd
Window::
GetActionFromNN_Device()
{
	Eigen::VectorXd state = mEnv->GetCharacter()->GetDevice()->GetState();
	py::array_t<float> state_np = py::array_t<float>(state.rows());
	py::buffer_info state_buf = state_np.request(true);
	float* dest = reinterpret_cast<float*>(state_buf.ptr);

	for(int i =0;i<state.rows();i++)
		dest[i] = state[i];

	py::object get_action_device;
	get_action_device = device_nn_module.attr("get_action");

	py::object temp = get_action_device(state_np);
	py::array_t<float> action_np = py::array_t<float>(temp);

	py::buffer_info action_buf = action_np.request(true);
	float* srcs = reinterpret_cast<float*>(action_buf.ptr);

	Eigen::VectorXd action(mEnv->GetCharacter()->GetDevice()->GetNumAction());
	for(int i=0;i<action.rows();i++)
		action[i] = srcs[i];

	return action;
}

py::array_t<float> toNumPyArray(const Eigen::VectorXd& vec)
{
	int n = vec.rows();
	py::array_t<float> array = py::array_t<float>(n);

	auto array_buf = array.request(true);
	float* dest = reinterpret_cast<float*>(array_buf.ptr);
	for(int i =0;i<n;i++)
	{
		dest[i] = vec[i];
	}

	return array;
}

Eigen::VectorXd
Window::
GetActionFromNN()
{
	Eigen::VectorXd state = mEnv->GetCharacter()->GetState();
	py::array_t<float> state_np = py::array_t<float>(state.rows());
	py::buffer_info state_buf = state_np.request(true);
	float* dest = reinterpret_cast<float*>(state_buf.ptr);

	for(int i =0;i<state.rows();i++)
		dest[i] = state[i];

	py::object apply;
	apply = rms_module.attr("apply_no_update");
	py::object state_np_tmp = apply(state_np);
	py::array_t<float> state_np_ = py::array_t<float>(state_np_tmp);

	py::object get_action;
	get_action = nn_module.attr("get_action");
	py::object temp = get_action(state_np_);
	py::array_t<float> action_np = py::array_t<float>(temp);

	py::buffer_info action_buf = action_np.request(true);
	float* srcs = reinterpret_cast<float*>(action_buf.ptr);

	Eigen::VectorXd action(mEnv->GetCharacter()->GetNumAction());
	for(int i=0;i<action.rows();i++)
		action[i] = srcs[i];

	return action;
}

Eigen::VectorXd
Window::
GetActivationFromNN(const Eigen::VectorXd& mt)
{
	if(!mMuscleNNLoaded)
	{
		mEnv->GetCharacter()->GetDesiredTorques();
		return Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
	}

	py::object get_activation = muscle_nn_module.attr("get_activation");
	mEnv->GetCharacter()->SetDesiredTorques();
	Eigen::VectorXd dt = mEnv->GetCharacter()->GetDesiredTorques();
	py::array_t<float> mt_np = toNumPyArray(mt);
	py::array_t<float> dt_np = toNumPyArray(dt);
	py::array_t<float> activation_np = get_activation(mt_np,dt_np);
	py::buffer_info activation_np_buf = activation_np.request(false);
	float* srcs = reinterpret_cast<float*>(activation_np_buf.ptr);

	Eigen::VectorXd activation(mEnv->GetCharacter()->GetMuscles().size());
	for(int i=0; i<activation.rows(); i++){
		activation[i] = srcs[i];
	}

	return activation;
}

}
