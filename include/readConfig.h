/*
 * readConfig.h
 *
 *  Created on: Apr 17, 2019
 *      Author: Michal Szutenberg
 */

#ifndef LANETRACKERAPP_READCONFIG_H_
#define LANETRACKERAPP_READCONFIG_H_

#include <string>
#include "Config.h"

using namespace std;

int readConfig(string path, LaneTracker::Config* cfg);



#endif /* LANETRACKERAPP_READCONFIG_H_ */
