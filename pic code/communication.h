#ifndef communication
#define communication

#define CS1 LATDbits.LATD9 // communication with the modulator
#define CS2 LATBbits.LATB8
//#define tt LATBbits.LATB13;

#define msg_length 10
#define pi 3.141592

void spi_init();
void spi_write(int input);
void calibrate();
void zero_sensors();
double mu_compute();
int spi_read(int n);

int out_val = 4095;
int p = 0, i;
int ACQUIRE_DATA = 1, SET_FB_CONTROL = 0, HAPTICS = 0, touching = 0;

char message[msg_length];

double NOR[10], LAT[10];
double nor, lat, norl, latl, norf = 0, latf = 0, norfl = 0, latfl = 0, wc = 2*150*pi;
double nor10 = 0, lat10 = 0, lat10l;
double Nc = 805, Lc = 548;
double mnor[100], mlat[100], nor_start = 0, lat_start = 0;
double mu = 0, t = 0, dt = .0001, lambda = 0, vol, mul = 0, mull = 0, dmu = 0, dmul = 0, dir, maxdmu = 0, maxmu = 0;
double Islip, Istick, Islide, sigma, delta;
int gen_var, u, l, psmode = 1, state = 0, old_state = 0, lock = 0;

#endif
