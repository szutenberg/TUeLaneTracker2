/*
 * polyFit.h
 *
 *  Created on: Mar 14, 2019
 *      Author: msz
 */

#ifndef TUELDT_POLYFIT_H_
#define TUELDT_POLYFIT_H_

#include<vector>

void polyFit(const std::vector<double> &xv, const std::vector<double> &yv, std::vector<double> &coeff, int order);


#endif /* TUELDT_POLYFIT_H_ */
