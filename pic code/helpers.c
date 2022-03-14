#include "NU32.h"
#include <xc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "interrupts.h"
#include "helpers.h"
#include "communication.h"
#include <time.h>

// Given an array of an arbitrary size, avg_of_array will take the average of the first n elements
double avg_of_array(double arr[], int lim){
  double total = 0;
  for (i=0;i<lim;i++){
    total += arr[i];
  }
  return total/lim;
}

double sgn(double a){
  if (a<0){
    return -1;
  }else{
    return 1;
  }
}

double abss(double a){
  if (a<0) {
    return -a;
  } else {
    return a;
  }
}

double F_function(double us, double uk, double l, double t, int type){
  switch(type){
    case 0:
    if ((us - t*l)>uk){
      return (us - t*l);
    } else {
      return uk;
    }
    break;
    case 1:
    return 2*(us - uk)/(1+exp(2*l*t/(us-uk))) + uk;
    break;
  }
}

/*
double nrand(){
  return sqrt(-2*log((double) rand()/RAND_MAX))*sin(2*pi*((double) rand()/RAND_MAX));
}
*/
