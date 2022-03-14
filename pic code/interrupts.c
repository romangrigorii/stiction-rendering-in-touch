#include <xc.h>          // Load the proper header for the processor
#include "NU32.h"
#include "interrupts.h"
#include <math.h>
#include "communication.h"
#include "helpers.h"

void interrupt_init(){
  PR2 = 3999;
  TMR2 = 0;
  T2CONbits.TCKPS = 1; // prescaler of 1
  T2CONbits.ON = 1;
  IPC2bits.T2IP = 1; // priority for timer 2
  IPC2bits.T2IS = 0; //
  IFS0bits.T2IF = 0; // clear interrupt flag
  IEC0bits.T2IE = 1; // enable interrupt
}

void __ISR(_TIMER_2_VECTOR, IPL1SOFT) Timer2Vel(void){
  if (ACQUIRE_DATA){
    latl = lat;
    norl = nor;
    LATBbits.LATB13 = 0;
    lat = (double) spi_read(0);
    nor = (double) spi_read(1);
    latfl = latf;
    norfl = norf;
    latf = (lat + latl - latfl*(1 - 1/(wc*dt)))/(1 + 1/(wc*dt));
    norf = ((nor + norl) - norfl*(1 - 1/(wc*dt)))/(1 + 1/(wc*dt));
    NOR[p] = norf;
    LAT[p] = latf;
    p++;
    LATFINV = 2;
    if (p>=10){ // this occurs every ~1kHz
      p = 0;
      nor10 = avg_of_array(NOR,10);
      lat10 = avg_of_array(LAT,10);
      if ((nor10-nor_start)>30){ // the screen us considered being touched when normal load exeeds .025 N
        touching = 1;
      } else {
        touching = 0;
      }
      if (touching && HAPTICS && lock == 0){// proceed if the subject is touching the display, haptic feedback is turned on, and the routine is not locked
        old_state = state;
        mul = mu;
        mu_compute();
        dmul = dmu;
        dmu = (mu - mul)/(dt*10);
        if (lock == 0){
          if (abss(mu)<sigma){
            state = 0;
          } else {
            if ((state == 0) && (abss(dmu) < (maxdmu*delta))){
              state = 1;
            }
            if ((state == 1) && ((abss(mu) - abss(mul))<=0)){
              state = 2;
            }
          }
        }
      }
      if (state == 0){
        if (maxdmu<abss(dmu)){
          maxdmu = abss(dmu);
        }
      }
      if (state == 2){
        maxdmu = 0;
      }
      if (old_state!=state){
        lock = 0;
      }
      else if (lock){
        l++;
        if (l>=2){
          lock = 0;
          l = 0;
        }
      }
      switch (state){
        case 0:
        out_val = Istick;
        break;
        case 1:
        out_val = Islip;
        break;
        case 2:
        out_val = Islide;
        break;
      }
      spi_write(out_val);
    }
  }
  LATBbits.LATB13 = 1;
  IFS0bits.T2IF = 0;
}
