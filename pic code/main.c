#include <xc.h>          // Load the proper header for the processor
#include "interrupts.h"
#include "NU32.h"
#include "communication.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"
char mode = 'a';

int main(void){
  //srand(time(NULL));
  __builtin_disable_interrupts();
  NU32_Startup();
  spi_init();
  interrupt_init();
  __builtin_enable_interrupts();
  NU32_Startup();
  NU32_ReadUART3(message, msg_length);
  zero_sensors();
  HAPTICS = 1;
  while(1){
    switch(mode){
      case 'a': // assign mode
      sprintf(message,"%s\r\n","home mode");
      NU32_WriteUART3(message);
      __builtin_enable_interrupts();
      NU32_ReadUART3(message, msg_length);
      sscanf(message,"%c",&mode);
      if (mode =='z' ||  mode == 'm' || mode == 'r'){
      }else{
        mode = 'a';
      }
    break;
    case 'm':
    __builtin_disable_interrupts();
    sprintf(message,"%s\r\n","parameter select mode");
    NU32_WriteUART3(message);
    NU32_ReadUART3(message, msg_length);
    sscanf(message,"%lf",&Istick);
    NU32_ReadUART3(message, msg_length);
    sscanf(message,"%lf",&Islip);
    NU32_ReadUART3(message, msg_length);
    sscanf(message,"%lf",&Islide);
    NU32_ReadUART3(message, msg_length);
    sscanf(message,"%lf",&sigma);
    NU32_ReadUART3(message, msg_length);
    sscanf(message,"%lf",&delta);
    __builtin_enable_interrupts();
    mode = 'a';
    break;
    case 'z': // zero the sensors
    zero_sensors();
    __builtin_disable_interrupts();
    sprintf(message,"%s\r\n","data zeroed");
    NU32_WriteUART3(message);
    __builtin_enable_interrupts();
    mode = 'a';
    break;
    case 'r':
    __builtin_disable_interrupts();
    sprintf(message,"\t%s%lf\t%s%lf\t%s%lf\t%s%lf\t%s%lf\r\n","Istick= ",Istick,"Islip = ",Islip, "Islide = ",Islide, "sigma = ",sigma,"delta = ",delta);
    NU32_WriteUART3(message);
    __builtin_enable_interrupts();
    mode = 'a';
    break;
  }
}
}
