#include <SigInt.h>
#include "FrameFeeder.h"
#include "StateMachine.h"
#include "boost/program_options.hpp"
#include "readConfig.h"

using namespace std;
namespace po = boost::program_options;


//Fucntion definitions
unique_ptr<FrameFeeder> createFrameFeeder(FrameSource srcMode, string srcString);



int main(int argc, char* argv[]) /**
	This is the entry point of the application.
	- Initialises the sigInit handler
	- Creates a stateMachine and spins it until user issues a quit signal through the sigInit handler.
	*/
{
	int lReturn 	= 0;

	FrameSource 	lFrameSource;
	std::string 	lSourceStr;
	string lConfigFileName;



	// Prsing command line options
	{
	  po::options_description	lDesc("Options");

	  lDesc.add_options()
	  ("help,h",
			"\t produces help message")

	  ("Mode,m",
			po::value<FrameSource>(&lFrameSource)->default_value(FrameSource::DIRECTORY),
			"\t selects frame input mode")

	  ("Source,s",
			po::value<string>(&lSourceStr)->default_value("DataSet"),
			"\t provides source configuration")

	  ("Config,c", po::value<string>(&lConfigFileName)->default_value(""),
			"\t yaml configuration file");



			;

	  po::variables_map vMap;
	  po::store(po::parse_command_line(argc, argv, lDesc), vMap);
	  po::notify(vMap);

	  if ( vMap.count("help") )
	  {
 	     cout << lDesc <<endl;
	     cout << "	Valid arguments for 'Mode': ";
	     cout << "["<<FrameSource::DIRECTORY<<" ";
	     cout <<FrameSource::STREAM<<" ";
	     cout <<FrameSource::GMSL<<"]";

	     cout <<endl<<endl<< "	Examples:"<<endl;
	     cout<< "	./TUeLaneTracker -m " << FrameSource::DIRECTORY << " -s " << "/home/DataSet -c Config.yaml \n";
	     cout<<endl<<endl;
	     lReturn = 1;
	  }

	} // End parsing command line options



	unique_ptr<LaneTracker::Config> lPtrConfig;
	if (lReturn == 0) //create Configuration
	{
	  lPtrConfig.reset(new LaneTracker::Config);
	  if(lPtrConfig == nullptr)
	  {
	    lReturn = -1;
	  }
	}

	if ((lReturn == 0) && (!lConfigFileName.empty()))
	{
		lReturn = readConfig(lConfigFileName, lPtrConfig.get());
	}

	unique_ptr<FrameFeeder> lPtrFeeder;
	if (lReturn == 0) //create FrameFeeder
	{
	  lPtrFeeder = createFrameFeeder(lFrameSource, lSourceStr);
	  if(lPtrFeeder == nullptr)
	  {
	    lReturn = -1;
	  }
	}


	shared_ptr<SigInt> lPtrSigInt;
	if(lReturn == 0) //create SigInt
	{
	  lPtrSigInt = make_shared<SigInt>();
	  if(lPtrSigInt->sStatus == SigStatus::FAILURE)
	  {
	    lReturn = -1;
	  }
	}


	unique_ptr<StateMachine> lPtrStateMachine;
	if (lReturn==0) //create StateMachine
	{
	  try
	  {
	       lPtrStateMachine.reset( new StateMachine( move(lPtrFeeder), *lPtrConfig.get() ) );
	  }
	  catch(const char* msg)
	  {
		cerr << "Failed to create the StateMachine!: " << msg << "\n";
		lReturn =-1;
	  }
	  catch(...)
	  {
		lReturn = -1;
	  }
	}


	States lPreviousState;
	if (lReturn ==0) //Get current State of the stateMachine
	{
//	  cout<<lPtrStateMachine->getCurrentState();
	  lPreviousState = lPtrStateMachine->getCurrentState();
	}



    if(lReturn == 0) //spin the stateMachine
    {
       uint64_t        lCyclesCount = 0;
       ProfilerLDT     lProfiler;
       StateMachine&   stateMachine = *lPtrStateMachine.get();

       while (stateMachine.getCurrentState() != States::DISPOSED)
	   {

	     lProfiler.start("StateMachine_Cycle");
	     if (lPtrSigInt->sStatus == SigStatus::STOP)
	      stateMachine.quit();

	     lReturn = stateMachine.spin();
	     lCyclesCount ++;

	     lProfiler.end();

	     if(lPreviousState != stateMachine.getCurrentState())
	     {
	       //cout<<endl<<stateMachine.getCurrentState();
	       //std::cout.flush();
	       lPreviousState = stateMachine.getCurrentState();
	     }
	     else if (lCyclesCount%100==0)
	     {
/*	       cout <<endl<<stateMachine.getCurrentState();
	       cout <<"state cycle-count = " << lCyclesCount<<"    Cycle-Time [Min, Avg, Max] : "
	       <<"[ "<<lProfiler.getMinTime("StateMachine_Cycle")<<" "
	       <<lProfiler.getAvgTime("StateMachine_Cycle")<<" "
	       <<lProfiler.getMaxTime("StateMachine_Cycle")<<" "
	       <<" ]";*/
	     }

	   }// End spinning
    }

    lPtrStateMachine.reset( nullptr);
	cerr << "The program ended with exit code " <<lReturn<<endl;
	return lReturn;
}



unique_ptr<FrameFeeder> createFrameFeeder(FrameSource srcMode, string srcString)
{
	unique_ptr<FrameFeeder>	lPtrFeeder;

	/** Create Image Feeder */
	try
	{
	  switch(srcMode)
	  {
            case DIRECTORY:
               lPtrFeeder=  unique_ptr<FrameFeeder>( new ImgStoreFeeder(srcString) );
               break;
            case STREAM:
              lPtrFeeder=  unique_ptr<FrameFeeder>( new StreamFeeder(srcString) );
              break;
            case GMSL:
              throw "NOT IMPLEMENTED";
              break;
	  }
	}
	catch(const char* msg)
	{
	    cerr << "Failed to create the FrameFeeder!: " << msg << "\n";
	    lPtrFeeder = nullptr;
	}
	catch (...)
	{
	    cerr << "Failed to create the FrameFeeder!: unknown exception\n";
	   lPtrFeeder = nullptr;
	}

	return lPtrFeeder;

}











#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

static void help()
{
    // print a welcome message, and the OpenCV version
    cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
            "Using OpenCV version " << CV_VERSION << endl;
    cout << "\nIt uses camera by default, but you can provide a path to video as an argument.\n";
    cout << "\nHot keys: \n"
            "\tESC - quit the program\n"
            "\tr - auto-initialize tracking\n"
            "\tc - delete all the points\n"
            "\tn - switch the \"night\" mode on/off\n"
            "To add/remove a feature point click it\n" << endl;
}

Point2f point;
bool addRemovePt = false;

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == EVENT_LBUTTONDOWN )
    {
        point = Point2f((float)x, (float)y);
        addRemovePt = true;
    }
}


vector<cv::String> parseSettings(string& srcStr)
{
   string lDelimiter = ",";
   size_t lPos	     =  0 ;
   vector<cv::String> mFiles;
   vector<string>  lTokens;
   while ( (lPos = srcStr.find(lDelimiter) ) != string::npos )
   {
     lTokens.push_back(srcStr.substr(0, lPos));
     srcStr.erase(0, lPos + lDelimiter.length() );
   }
   lTokens.push_back(srcStr);  //push_back the last substring too.

   string mFolder   = lTokens[0];

    glob(mFolder, mFiles);
    sort(mFiles.begin(), mFiles.end());
    for (size_t i = 0; i < mFiles.size(); i++)
    {
    	const int EXT_LEN = strlen(".png");
    	cv::String extension = mFiles[i].substr(mFiles[i].length() - EXT_LEN);
    	if (extension != ".png")
    	{
    		mFiles.erase(mFiles.begin() + i);
    		i--;
    	}
    }

   return mFiles;

}


int main2( int argc, char** argv )
{
    VideoCapture cap;
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size subPixWinSize(10,10), winSize(31,31);

    const int MAX_COUNT = 20000;
    bool needToInit = true;
    bool nightMode = false;

    help();
    cv::CommandLineParser parser(argc, argv, "{@input|0|}");
    string input = parser.get<string>("@input");

    if( input.size() == 1 && isdigit(input[0]) )
        cap.open(input[0] - '0');
    else
        cap.open(input);

    if( !cap.isOpened() )
    {
        cout << "Could not initialize capturing...\n";
       // return 0;
    }

    namedWindow( "LK Demo", 1 );
    setMouseCallback( "LK Demo", onMouse, 0 );

    Mat gray, prevGray, image, frame;
    vector<Point2f> points[10];

	unique_ptr<FrameFeeder> lPtrFeeder;
	// lPtrFeeder = createFrameFeeder(FrameSource::DIRECTORY, "/data/benchmark/dataset/p19017");
	// lPtrFeeder->Paused.store(false);
	 string path = "/data/benchmark/dataset/p19017";
	 vector<cv::String> mFiles = parseSettings(path);

	 Mat t[mFiles.size()];

	 for (int i = 0; i < mFiles.size(); i++)
	 {
		 Mat tmp = imread(mFiles[i]);
		   cv::Rect lROI;

		  int  lRowIndex = 260;
		   lROI = cv::Rect(0,lRowIndex,tmp.cols, tmp.rows-lRowIndex);
		  tmp(lROI).copyTo(t[i]);

	 }

    for(int i = 0; i < mFiles.size(); i++)
    {
    	frame = t[i];
        //cap >> frame;
        if( frame.empty() )
            break;

        frame.copyTo(image);
        cvtColor(image, gray, COLOR_BGR2GRAY);

        if( nightMode )
            image = Scalar::all(0);

        if (i%10 == 0) needToInit = true;

        if( needToInit )
        {
            // automatic initialization
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 5, Mat(), 3, 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);
            addRemovePt = false;
        }
        else if( !points[0].empty() )
        {
            vector<uchar> status;
            vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, k;
            for( i = k = 0; i < points[1].size(); i++ )
            {
                if( addRemovePt )
                {
                    if( norm(point - points[1][i]) <= 5 )
                    {
                        addRemovePt = false;
                        continue;
                    }
                }

                if( !status[i] )
                    continue;

                points[1][k++] = points[1][i];

                line (image, points[0][i], points[1][i], Scalar(0, 0, 255), 3);

                circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
            points[1].resize(k);
        }

        if( addRemovePt && points[1].size() < (size_t)MAX_COUNT )
        {
            vector<Point2f> tmp;
            tmp.push_back(point);
            cornerSubPix( gray, tmp, winSize, Size(-1,-1), termcrit);
            points[1].push_back(tmp[0]);
            addRemovePt = false;
        }

        needToInit = false;
        imshow("LK Demo", image);

        char c = (char)waitKey(10);
        if( c == 27 )
            break;
        switch( c )
        {
        case 'r':
            needToInit = true;
            break;
        case 'c':
            points[0].clear();
            points[1].clear();
            break;
        case 'n':
            nightMode = !nightMode;
            break;
        case ' ':
        	c = (char)waitKey(0);
        	break;
        }

        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
       // usleep(100000);
    }

    return 0;
}
