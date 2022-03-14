#include <xc.h>          // Load the proper header for the processor
#include "interrupts.h"
#include "NU32.h"
#include "communication.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"


void spi_init(){
  TRISDbits.TRISD9 = 0;
  TRISBbits.TRISB8 = 0;
  TRISBbits.TRISB13 = 0;
  CS1 = 1;
  CS2 = 1;
  SPI3CONbits.MSTEN = 1;                                                        // master mode
  SPI3CONbits.MODE16 = 1;                                                       // sets the communication to be 16 bit
  SPI3BRG = 19;                                                                  // SPI3BRG = 80,000,000/(2*160,000) - 1;
  SPI3STATbits.SPIROV = 0;
  SPI3CONbits.CKE = 1;
  SPI3CONbits.SMP = 1;
  SPI3CONbits.CKP = 1;
  SPI3CONbits.ON = 1;
  SPI4CON = 39;
  SPI4BUF;
  SPI4STATbits.SPIROV = 0;
  TRISBbits.TRISB8 = 0;
  CS2 = 1;
  SPI4CONbits.MSTEN = 1;
  SPI4CONbits.MODE32 = 0;
  SPI4CONbits.MODE16 = 0;
  SPI4BRG = 19;                                                                // SPI4BRG = 80,000,000/(2*320,000) - 1;
  SPI4CONbits.CKE = 1;
  SPI4CONbits.ON = 1;
}


int spi_read(int n){
  int read1 = 0, read2 = 0;
  CS2 = 0;
  switch (n){
    case 0:
    SPI4BUF = 0b10001111;                                                    // real normal force ch0 and ch1
    break;
    case 1:
    SPI4BUF = 0b11001111;                                                   // reads lateral force from ch0 and ch1
    break;
  }
  if ((n==0) || (n==1)){
    while(!SPI4STATbits.SPIRBF){
    }
    read1 = SPI4BUF;
    SPI4BUF = 0x00;
    while(!SPI4STATbits.SPIRBF){
    }
    read1 = SPI4BUF;
    SPI4BUF = 0x00;
    while(!SPI4STATbits.SPIRBF){
    }
    read2 = SPI4BUF;
  }
  CS2 = 1;
  return ((read1<<5) + (read2>>3));
}


void spi_write(int input){
    CS1 = 1;
    SPI3BUF = input>>1;
    while(!SPI3STATbits.SPIRBF){
    }
    SPI3BUF;
    CS1 = 0;
  }

  void zero_sensors(){
    for (u=0;u<100;u++){
      _CP0_SET_COUNT(0);
      while (_CP0_GET_COUNT()<4000){
      }
      mnor[u] = nor10;
      mlat[u] = lat10;
    }
    nor_start = avg_of_array(mnor,100);
    lat_start = avg_of_array(mlat,100);
  }


  double mu_compute(){
    if (abss((nor10 - nor_start)/Nc) < .01){ // avoiding singularities
      mu = 0;
    } else {
      mu = (lat10 - lat_start)/(nor10 - nor_start)*Nc/Lc;
    }
    return mu;
  }
