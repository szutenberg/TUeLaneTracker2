#include <SigInt.h>
#include "FrameFeeder.h"
#include "StateMachine.h"
#include "boost/program_options.hpp"

using namespace std;
namespace po = boost::program_options;

//Function definitions
unique_ptr<FrameFeeder> createFrameFeeder(FrameSource srcMode,
		string srcString);

int readConfig(string path, LaneTracker::Config* config);

int main(int argc, char* argv[]) /**
 This is the entry point of the application.
 - Initializes the sigInit handler
 - Creates a stateMachine and spins it until user issues a quit signal through the sigInit handler.
 */
{
	int lReturn = 0;

	FrameSource lFrameSource;
	string lSourceStr;
	string lConfigFileName;

	unique_ptr<LaneTracker::Config> lPtrConfig;
	if (lReturn == 0) //create Configuration
			{
		lPtrConfig.reset(new LaneTracker::Config);
		if (lPtrConfig == nullptr) {
			lReturn = -1;
		}
	}

	// Parsing command line options
	{
		po::options_description lDesc("Options");

		lDesc.add_options()("help,h", "\t produces help message")

		("Mode,m",
				po::value<FrameSource>(&lFrameSource)->default_value(
						FrameSource::DIRECTORY), "\t selects frame input mode")

		("Source,s", po::value<string>(&lSourceStr)->default_value("DataSet"),
				"\t provides source configuration")

		("Config,c", po::value<string>(&lConfigFileName)->default_value(""),
				"\t yaml configuration file");

		po::variables_map vMap;
		po::store(po::parse_command_line(argc, argv, lDesc), vMap);
		po::notify(vMap);

		if (vMap.count("help")) {
			cout << lDesc << endl;
			cout << "	Valid arguments for 'Mode': ";
			cout << "[" << FrameSource::DIRECTORY << " ";
			cout << FrameSource::STREAM << " ";
			cout << FrameSource::GMSL << "]";

			cout << endl << endl << "	Examples:" << endl;
			cout << "	./TUeLaneTracker -m " << FrameSource::DIRECTORY << " -s "
					<< "/home/DataSet -c Config.yaml" << endl;
			cout << endl << endl;
			lReturn = 1;
		}

	} // End parsing command line options

	if ((lReturn == 0) && (!lConfigFileName.empty()))
	{
		lReturn = readConfig(lConfigFileName, lPtrConfig.get());
	}

	// TODO: explain it in help
	if (lSourceStr.find(".mp4") != string::npos) lFrameSource = FrameSource::STREAM;

	unique_ptr<FrameFeeder> lPtrFeeder;
	if (lReturn == 0) //create FrameFeeder
			{
		lPtrFeeder = createFrameFeeder(lFrameSource, lSourceStr);
		if (lPtrFeeder == nullptr) {
			lReturn = -1;
		}
	}

	shared_ptr<SigInt> lPtrSigInt;
	if (lReturn == 0) //create SigInt
			{
		lPtrSigInt = make_shared<SigInt>();
		if (lPtrSigInt->sStatus == SigStatus::FAILURE) {
			lReturn = -1;
		}
	}

	unique_ptr<StateMachine> lPtrStateMachine;
	if (lReturn == 0) //create StateMachine
			{
		std::cout << endl << endl;
		std::cout << "******************************" << std::endl;
		std::cout << " Press Ctrl + c to terminate." << std::endl;
		std::cout << "******************************" << std::endl;

		try {
			lPtrStateMachine.reset(
					new StateMachine(move(lPtrFeeder), *lPtrConfig.get()));
		} catch (const char* msg) {
			cout << "******************************" << endl
					<< "Failed to create the StateMachine!" << endl << msg
					<< endl << "******************************" << endl << endl;
			lReturn = -1;
		} catch (...) {
			lReturn = -1;
		}
	}

	States lPreviousState;
	if (lReturn == 0) //Get current State of the stateMachine
			{
		cout << lPtrStateMachine->getCurrentState();
		lPreviousState = lPtrStateMachine->getCurrentState();
	}

	if (lReturn == 0) //spin the stateMachine
			{
		uint64_t lCyclesCount = 0;
		ProfilerLDT lProfiler;
		StateMachine& stateMachine = *lPtrStateMachine.get();

		while (stateMachine.getCurrentState() != States::DISPOSED) {

			lProfiler.start("StateMachine_Cycle");
			if (lPtrSigInt->sStatus == SigStatus::STOP)
				stateMachine.quit();

			lReturn = stateMachine.spin();
			lCyclesCount++;

			lProfiler.end();

			if (lPreviousState != stateMachine.getCurrentState()) {
				cout << endl << stateMachine.getCurrentState();
				std::cout.flush();
				lPreviousState = stateMachine.getCurrentState();
			} else if (lCyclesCount % 100 == 0) {
				cout << endl << stateMachine.getCurrentState();
				cout << "state cycle-count = " << lCyclesCount
						<< "    Cycle-Time [Min, Avg, Max] : " << "[ "
						<< lProfiler.getMinTime("StateMachine_Cycle") << " "
						<< lProfiler.getAvgTime("StateMachine_Cycle") << " "
						<< lProfiler.getMaxTime("StateMachine_Cycle") << " "
						<< " ]";
			}

		} // End spinning
	}

	lPtrStateMachine.reset(nullptr);
	cout << endl << "The program ended with exit code " << lReturn << endl;
	return lReturn;
}

unique_ptr<FrameFeeder> createFrameFeeder(FrameSource srcMode,
		string srcString) {
	unique_ptr<FrameFeeder> lPtrFeeder;

	/** Create Image Feeder */
	try {
		switch (srcMode) {
		case DIRECTORY:
			lPtrFeeder = unique_ptr<FrameFeeder>(new ImgStoreFeeder(srcString));
			break;
		case STREAM:
			lPtrFeeder = unique_ptr<FrameFeeder>(new StreamFeeder(srcString));
			break;
		case GMSL:
			throw "NOT IMPLEMENTED";
			break;
		}
	} catch (const char* msg) {
		cout << "******************************" << endl
				<< "Failed to create the FrameFeeder!" << endl << msg << endl
				<< "******************************" << endl << endl;
		lPtrFeeder = nullptr;
	} catch (...) {
		cout << "******************************" << endl
				<< "Failed to create the FrameFeeder!" << endl
				<< "Unknown exception" << endl
				<< "******************************" << endl << endl;
		lPtrFeeder = nullptr;
	}

	return lPtrFeeder;
}


int readConfig(std::string path, LaneTracker::Config* cfg)
{
	struct stat buf;
	int statResult = stat(path.c_str(),&buf);
	if (statResult != 0)
	{
		cout << "File " << path << " not found!" << endl;
		return 2;
	}

	FileStorage fs(path, FileStorage::READ);

	cfg->lane_avg_width = fs["lane_avg_width"];
	cfg->lane_std_width = fs["lane_std_width"];
	cfg->lane_min_width = fs["lane_min_width"];
	cfg->lane_max_width = fs["lane_max_width"];
	cfg->lane_marker_width = fs["lane_marker_width"];
	cfg->cam_res_v = fs["cam_res_v"];
	cfg->cam_res_h = fs["cam_res_h"];
	cfg->cam_fx = fs["cam_fx"];
	cfg->cam_fy = fs["cam_fy"];
	cfg->cam_cx = fs["cam_cx"];
	cfg->cam_cy = fs["cam_cy"];
	cfg->cam_pitch = fs["cam_pitch"];
	cfg->cam_yaw = fs["cam_yaw"];
	cfg->cam_height = fs["cam_height"];
	cfg->cam_lateral_offset = fs["cam_lateral_offset"];
	cfg->base_line_IBCS = fs["base_line_IBCS"];
	cfg->purview_line_IBCS = fs["purview_line_IBCS"];
	cfg->step_lane_filter_cm = fs["step_lane_filter_cm"];
	cfg->step_vp_filter = fs["step_vp_filter"];
	cfg->vp_range_ver = fs["vp_range_ver"];
	cfg->vp_range_hor = fs["vp_range_hor"];
	cfg->buffer_count = fs["buffer_count"];
	cfg->display_graphics = ((int)fs["display_graphics"]) != 0;
	fs.release();

	return 0;
}
