#include <iostream>
#include <fstream>
#include <utility> 
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <numbers>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <nanobind/eigen/dense.h>
#include <nanobind/eigen/sparse.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "eigen_defines.h"

namespace py = nanobind;

const real pi_real = std::numbers::pi_v<real>;




class c_lubrication
{
  private:
  void SetMemberData(std::string fname, std::vector< std::vector<real> >& vec_11, 
        std::vector< std::vector<real> >& vec_12, std::vector<real>& x);
  void SetMemberDataWall(std::string fname, std::vector< std::vector<real> >& vec, std::vector<real>& x, bool reverse);
  std::vector< std::vector<real> > mob_scalars_WS_11, mob_scalars_WS_12;
  std::vector<real> WS_x;
  std::vector< std::vector<real> > mob_scalars_JO_11, mob_scalars_JO_12;
  std::vector<real> JO_x;
  std::vector< std::vector<real> > mob_scalars_MB_11, mob_scalars_MB_12;
  std::vector<real> MB_x;
  std::vector< std::vector<real> > mob_scalars_wall_2562;
  std::vector<real> Wall_2562_x;
  std::vector< std::vector<real> > mob_scalars_wall_MB;
  std::vector<real> Wall_MB_x;
  int FindNearestIndexLower(real r_norm, std::vector<real>& x);
  real LinearInterp(real r_norm, real xL, real xR, real yL, real yR);
  void ResistMatrix(real r_norm, real mob_factor[3], Vector3 r_hat, Matrix& R, bool inv, std::vector<real>& x, 
        const std::vector< std::vector<real> >& vec_11, const std::vector< std::vector<real> >& vec_12);
  void ATResistMatrix(real r_norm, real mob_factor[3], Vector3 r_hat, Matrix& R);
  Matrix WallResistMatrix(real r_norm, real mob_factor[3], std::vector< real >& x, 
          const std::vector< std::vector< real > >& vec);
  Matrix  ResistPairLub(real r_norm, real mob_factor[3], Vector3 r_hat);
  Matrix WallResistMatrixBlob(real r_norm, real mob_factor[3], std::vector< real >& x, 
          const std::vector< std::vector< real > >& vec);
  Matrix ResistPairBlob(real r_norm, real mob_factor[3], Vector3 r_hat);

  std::vector<Eigen::Triplet<real>> makeTriplets(const std::vector<real>& data, 
                                    const std::vector<int>& rows, const std::vector<int>& cols);

  real tolerance = 10e-4;
  real cutoff_wall = 1.0e10;

  public:
  void Set_R_Mats(Matrix r_vecs, IMatrix neighbors, real a, real eta, real cutoff, real cutoff_wall, 
        std::vector<real> periodic_length, bool Sup_if_true);
  void ResistPairLub_py(real r_norm, real a, real eta, std::vector<real> r_hat);
  void ResistCOO(Matrix r_vectors, IMatrix n_list, real a, real eta, real cutoff, real wall_cutoff, 
        std::vector<real> periodic_length, bool Sup_if_true, std::vector<real>& data, std::vector<int>& rows, 
        std::vector<int>& cols);
  void ResistCOO_wall(Matrix r_vectors, real a, real eta, real wall_cutoff, std::vector<real> periodic_length, 
        bool Sup_if_true, std::vector<real>& data, std::vector<int>& rows, std::vector<int>& cols);
  real debye_cut;
  c_lubrication(real d_cut);
  SpMatrix lub_print();

  SpMatrix R_blob;
  SpMatrix R_lub;
  SpMatrix Delta_R;
};

c_lubrication::c_lubrication(real d_cut)
{
  Py_Initialize();
  debye_cut = d_cut;
  std::string base_dir = __FILE__;
  SetMemberData(base_dir+"/Resistance_Coefs/mob_scalars_WS.txt",mob_scalars_WS_11,mob_scalars_WS_12,WS_x);
  SetMemberData(base_dir+"/Resistance_Coefs/res_scalars_JO.txt",mob_scalars_JO_11,mob_scalars_JO_12,JO_x);
  SetMemberDataWall(base_dir+"/Resistance_Coefs/mob_scalars_wall_MB_2562_eig_thresh.txt",mob_scalars_wall_2562,Wall_2562_x,true);
  SetMemberData(base_dir+"/Resistance_Coefs/res_scalars_MB_1.txt",mob_scalars_MB_11,mob_scalars_MB_12,MB_x);
  SetMemberDataWall(base_dir+"/Resistance_Coefs/res_scalars_wall_MB.txt",mob_scalars_wall_MB,Wall_MB_x,false);
}

void c_lubrication::SetMemberData(std::string fname, std::vector< std::vector<real> >& vec_11, 
                    std::vector< std::vector<real> >& vec_12, std::vector<real>& x)
{ 
  std::ifstream ifs(fname);
  real tempval;
  std::vector<real> tempv;
  
  if (!ifs.fail())
  {
    int p = 0;
    int c = -1;
    while(!ifs.eof())
    {
      c++;
      ifs >> tempval;
      tempv.push_back(tempval);
      if(c == 5)
      {
	p++;
	c=-1;
	if(p % 2){
	  vec_11.push_back(tempv);
	}
	else{
	  vec_12.push_back(tempv);
	}
	tempv.clear();
      }
    }
    ifs.close();
  }
  
  int k = 0;
  for (auto row : vec_11) {
    k++;
    x.push_back(row[0]);
  }

}


void c_lubrication::SetMemberDataWall(std::string fname, std::vector< std::vector<real> >& vec, 
                    std::vector<real>& x, bool reverse)
{ 
  std::ifstream ifs(fname);
  real tempval;
  std::vector<real> tempv;
  
  if (!ifs.fail())
  {
    int c = -1;
    while(!ifs.eof())
    {
      c++;
      ifs >> tempval;
      tempv.push_back(tempval);
      if(c == 5)
      {
	c=-1;
	if(reverse)
	{
	  vec.insert(vec.begin(), tempv);
	}
	else
	{
	  vec.push_back(tempv);
	}
	tempv.clear();
      }
    }
    ifs.close();
  }
  
  int k = 0;
  for (auto row : vec) {
    k++;
    x.push_back(row[0]);
  }

}

int c_lubrication::FindNearestIndexLower(real r_norm, std::vector<real>& x)
{
    // TODO: should make x a const vector but then distance fails
    // dunno what to do 
    std::vector<real>::iterator before, after, it;
    before = std::lower_bound(x.begin(), x.end(), r_norm);
    if(before == x.begin()){return -1;}
    if(before == x.end()){return x.size()-1;}
    
    after = before;
    --before;

    return std::distance(x.begin(),before);
    
}

real c_lubrication::LinearInterp(real r_norm, real xL, real xR, real yL, real yR)
{
    if(r_norm < xL || r_norm > xR){std::cout << "error in linear interp." << std::endl; return 1e100;}
    real dydx = ( yR - yL ) / ( xR - xL ); 
    return yL + dydx * ( r_norm - xL ); 
}

void c_lubrication::ResistMatrix(real r_norm, real mob_factor[3], Vector3 r_hat, Matrix& R, bool inv, 
                  std::vector< real >& x, const std::vector< std::vector< real > >& vec_11, 
                  const std::vector< std::vector< real > >& vec_12)
{
  
    real X11A, Y11A, Y11B, X11C, Y11C; 
    real X12A, Y12A, Y12B, X12C, Y12C;
    
    int Ind = FindNearestIndexLower(r_norm, x);
    
    if(Ind == -1 || Ind == x.size()-1)
    {
      int edge = (Ind == -1) ? 0 : (x.size()-1);
      X11A = vec_11[edge][1];
      Y11A = vec_11[edge][2];
      Y11B = vec_11[edge][3];
      X11C = vec_11[edge][4];
      Y11C = vec_11[edge][5]; 
      
      X12A = vec_12[edge][1]; 
      Y12A = vec_12[edge][2];
      Y12B = vec_12[edge][3];
      X12C = vec_12[edge][4];
      Y12C = vec_12[edge][5];
    }
    else
    {
      real a_11[5], a_12[5];
      real xL, yL11, yL12, xR, yR11, yR12;
      
      for(int i = 0; i < 5; i++){
	xL = x[Ind]; xR = x[Ind+1];
	yL11 = vec_11[Ind][i+1], yR11 = vec_11[Ind+1][i+1];
	yL12 = vec_12[Ind][i+1], yR12 = vec_12[Ind+1][i+1];
	a_11[i] = LinearInterp(r_norm, xL, xR, yL11, yR11);
	a_12[i] = LinearInterp(r_norm, xL, xR, yL12, yR12);
      }
      
      X11A = a_11[0];
      Y11A = a_11[1];
      Y11B = a_11[2];
      X11C = a_11[3];
      Y11C = a_11[4]; 
      
      X12A = a_12[0]; 
      Y12A = a_12[1];
      Y12B = a_12[2];
      X12C = a_12[3];
      Y12C = a_12[4];
    }
    
    Matrix3 squeezeMat = r_hat * r_hat.transpose();
    Matrix3 Eye;
    Eye.setIdentity(3,3);
    Matrix3 shearMat = Eye - squeezeMat;
    Matrix3 vortMat;
    vortMat << 0.0, r_hat[2], -r_hat[1],
               -r_hat[2], 0.0, r_hat[0],
               r_hat[1], -r_hat[0], 0.0;

    vortMat *= -1;
	       
	       
	       
    R.block<3,3>(0,0) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
    R.block<3,3>(0,3) = -mob_factor[1]*(Y11B*vortMat); // neg
    R.block<3,3>(0,6) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
    R.block<3,3>(0,9) = mob_factor[1]*(Y12B*vortMat); 
    
    R.block<3,3>(3,0) = mob_factor[1]*(Y11B*vortMat);
    R.block<3,3>(3,3) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
    R.block<3,3>(3,6) = mob_factor[1]*(Y12B*vortMat);
    R.block<3,3>(3,9) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
    
    R.block<3,3>(6,0) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
    R.block<3,3>(6,3) = -mob_factor[1]*(Y12B*vortMat); // neg
    R.block<3,3>(6,6) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
    R.block<3,3>(6,9) = mob_factor[1]*(Y11B*vortMat);
    
    R.block<3,3>(9,0) = -mob_factor[1]*(Y12B*vortMat); // neg
    R.block<3,3>(9,3) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
    R.block<3,3>(9,6) = -mob_factor[1]*(Y11B*vortMat); // neg
    R.block<3,3>(9,9) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
    
    if(inv){
      R = R.inverse();
    }
    
}

void c_lubrication::ATResistMatrix(real r_norm, real mob_factor[3], Vector3 r_hat, Matrix& R)
{
  
    real X11A, Y11A, Y11B, X11C, Y11C; 
    real X12A, Y12A, Y12B, X12C, Y12C;
    
    real epsilon = r_norm-2.0;
      
    X11A = 0.995419E0+(0.25E0)*(1.0/epsilon)+(0.225E0)*log((1.0/epsilon))+(0.267857E-1)*epsilon*log((1.0/epsilon));
    X12A = (-0.350153E0)+(-0.25E0)*(1.0/epsilon)+(-0.225E0)*log((1.0/epsilon))+(-0.267857E-1)*epsilon*log((1.0/epsilon));
    Y11A = 0.998317E0+(0.166667E0)*log((1.0/epsilon));
    Y12A = (-0.273652E0)+(-0.166667E0)*log((1.0/epsilon));
    Y11B = (-0.666667E0)*(0.23892E0+(-0.25E0)*log((1.0/epsilon))+(-0.125E0)*epsilon*log((1.0/epsilon)));
    Y12B = (-0.666667E0)*((-0.162268E-2)+(0.25E0)*log((1.0/epsilon))+(0.125E0)*epsilon*log((1.0/epsilon)));
    X11C = 0.133333E1*(0.10518E1+(-0.125E0)*epsilon*log((1.0/epsilon)));
    X12C = 0.133333E1*((-0.150257E0)+(0.125E0)*epsilon*log((1.0/epsilon)));
    Y11C = 0.133333E1*(0.702834E0+(0.2E0)*log((1.0/epsilon))+(0.188E0)*epsilon*log((1.0/epsilon)));
    Y12C = 0.133333E1*((-0.27464E-1)+(0.5E-1)*log((1.0/epsilon))+(0.62E-1)*epsilon*log((1.0/epsilon)));
    
    Matrix3 squeezeMat = r_hat * r_hat.transpose();
    Matrix3 Eye;
    Eye.setIdentity(3,3);
    Matrix3 shearMat = Eye - squeezeMat;
    Matrix3 vortMat;
    vortMat << 0.0, r_hat[2], -r_hat[1],
               -r_hat[2], 0.0, r_hat[0],
               r_hat[1], -r_hat[0], 0.0;

    vortMat *= -1;
	       
	       
	       
    R.block<3,3>(0,0) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
    R.block<3,3>(0,3) = -mob_factor[1]*(Y11B*vortMat); // neg
    R.block<3,3>(0,6) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
    R.block<3,3>(0,9) = mob_factor[1]*(Y12B*vortMat); 
    
    R.block<3,3>(3,0) = mob_factor[1]*(Y11B*vortMat);
    R.block<3,3>(3,3) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
    R.block<3,3>(3,6) = mob_factor[1]*(Y12B*vortMat);
    R.block<3,3>(3,9) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
    
    R.block<3,3>(6,0) = mob_factor[0]*(X12A*squeezeMat + Y12A*shearMat);
    R.block<3,3>(6,3) = -mob_factor[1]*(Y12B*vortMat); // neg
    R.block<3,3>(6,6) = mob_factor[0]*(X11A*squeezeMat + Y11A*shearMat);
    R.block<3,3>(6,9) = mob_factor[1]*(Y11B*vortMat);
    
    R.block<3,3>(9,0) = -mob_factor[1]*(Y12B*vortMat); // neg
    R.block<3,3>(9,3) = mob_factor[2]*(X12C*squeezeMat + Y12C*shearMat);
    R.block<3,3>(9,6) = -mob_factor[1]*(Y11B*vortMat); // neg
    R.block<3,3>(9,9) = mob_factor[2]*(X11C*squeezeMat + Y11C*shearMat);
    
}

Matrix c_lubrication::WallResistMatrix(real r_norm, real mob_factor[3], std::vector< real >& x, 
                      const std::vector< std::vector< real > >& vec)
{
  
    real Xa, Ya, Yb, Xc, Yc; 
    real Xa_asym, Ya_asym, Yb_asym, Xc_asym, Yc_asym; 
    real Xa_cutoff, Ya_cutoff, Yb_cutoff, Xc_cutoff, Yc_cutoff; 
    real RXa, RYa, RYb, RXc, RYc;
    real epsilon = r_norm-1.0;
    
    real tanh_fact = 1.0;
    if(epsilon < debye_cut) // epsilon < 0.1*debye_cut
    {
      epsilon = debye_cut;
      //tanh_fact = (1.0+tanh((epsilon+5*debye_cut)/2/debye_cut))*0.5;
      //epsilon = fmin(fabs(epsilon),debye_cut); //debye_cut;
      //if(epsilon < 0.1*debye_cut){epsilon = 0.1*debye_cut;}
      r_norm = 1.0+epsilon; //1.0+debye_cut;
    }
      
    int Ind = FindNearestIndexLower(r_norm, x);
    if(Ind == -1)
    {
      Xa = vec[0][1];
      Ya = vec[0][2];
      Yb = vec[0][3];
      Xc = vec[0][4];
      Yc = vec[0][5]; 
    }
    else if(Ind == x.size()-1)
    {
      Xa = 1.0 - (9.0/8.0)*(1.0/r_norm);
      Ya = 1.0 - (9.0/16.0)*(1.0/r_norm);
      Yb = 0.0;
      Xc = 0.75;
      Yc = 0.75;
    }
    else
    {
      real a[5];
      real xL, yL, xR, yR;
      
      for(int i = 0; i < 5; i++){
	xL = x[Ind]; xR = x[Ind+1];
	yL = vec[Ind][i+1], yR = vec[Ind+1][i+1];
	a[i] = LinearInterp(r_norm, xL, xR, yL, yR);
      }
      
      Xa = a[0];
      Ya = a[1];
      Yb = a[2];
      Xc = a[3];
      Yc = a[4]; 
    }
    
    
    
    Xa_asym = 1.0/epsilon - (1.0/5.0)*log(epsilon) + 0.971280;
    Ya_asym = -(8.0/15.0)*log(epsilon) + 0.9588;
    Yb_asym = -(-(1.0/10.0)*log(epsilon)-0.1895) - 0.4576*epsilon;
    Yb_asym *= 4./3.;
    Xc_asym = 1.2020569 - 3.0*(pi_real*pi_real/6.0-1.0)*epsilon;
    Xc_asym *= 4./3.;
    Yc_asym = -2.0/5.0*log(epsilon) + 0.3817 + 1.4578*epsilon;
    Yc_asym *= 4./3.;
    
    Xa_cutoff = 1.+0.1;
    Ya_cutoff = 1.+0.01;
    Yb_cutoff = 1.+0.1;
    Xc_cutoff = 1.+0.01;
    Yc_cutoff = 1.+0.1;
    
    real denom = Ya*Yc - Yb*Yb;
    RXa = 1.0/Xa;
    RYa = Yc/denom;
    RYb = -Yb/denom;
    RXc = 1.0/Xc;
    RYc = Ya/denom; 
    
    Xa = (r_norm > Xa_cutoff) ? RXa : Xa_asym;
    Ya = (r_norm > Ya_cutoff) ? RYa : Ya_asym;
    Yb = (r_norm > Yb_cutoff) ? RYb : Yb_asym;
    Xc = (r_norm > Xc_cutoff) ? RXc : Xc_asym;
    Yc = (r_norm > Yc_cutoff) ? RYc : Yc_asym;
    
    real XcPlus = fmax((Xc-4.0/3.0),0.0);
    real YcPlus = fmax((Yc-4.0/3.0),0.0);
    
    Matrix R(6,6);
    R << mob_factor[0]*(Ya-1.), 0, 0, 0, mob_factor[1]*Yb, 0,
	 0, mob_factor[0]*(Ya-1.), 0, -mob_factor[1]*Yb, 0, 0,
	 0, 0, mob_factor[0]*(Xa-1.), 0, 0, 0,
	 0, -mob_factor[1]*Yb, 0, mob_factor[2]*YcPlus, 0, 0,
	 mob_factor[1]*Yb, 0, 0, 0, mob_factor[2]*YcPlus, 0,
	 0, 0, 0, 0, 0, mob_factor[2]*XcPlus;
         

    if(fabs(tanh_fact-1.0) > 1e-10)
    {
      R *= tanh_fact;
    }
	 
    return R;
}


Matrix c_lubrication::WallResistMatrixBlob(real r_norm, real mob_factor[3], std::vector< real >& x, 
                      const std::vector< std::vector< real > >& vec)
{
  
    real Xa, Ya, Yb, Xc, Yc; 
    
    real epsilon = r_norm-1.0;
    real tanh_fact = 1.0;
    if(epsilon < debye_cut) // epsilon < 0.1*debye_cut
    {
      epsilon = debye_cut;
      //tanh_fact = (1.0+tanh((epsilon+5*debye_cut)/2/debye_cut))*0.5;
      //epsilon = fmin(fabs(epsilon),debye_cut); //debye_cut;
      //if(epsilon < 0.1*debye_cut){epsilon = 0.1*debye_cut;}
      r_norm = 1.0+epsilon; //1.0+debye_cut;
    }
      
    int Ind = FindNearestIndexLower(r_norm, x);
    if(Ind == -1)
    {
      Xa = vec[0][1];
      Ya = vec[0][2];
      Yb = vec[0][3];
      Xc = vec[0][4];
      Yc = vec[0][5]; 
    }
    else if(Ind == x.size()-1)
    {
      Xa = 1.0/(1.0 - (9.0/8.0)*(1.0/r_norm));
      Ya = 1.0/(1.0 - (9.0/16.0)*(1.0/r_norm));
      Yb = 0.0;
      Xc = 1.0/0.75;
      Yc = 1.0/0.75;
    }
    else
    {
      real a[5];
      real xL, yL, xR, yR;
      
      for(int i = 0; i < 5; i++){
	xL = x[Ind]; xR = x[Ind+1];
	yL = vec[Ind][i+1], yR = vec[Ind+1][i+1];
	a[i] = LinearInterp(r_norm, xL, xR, yL, yR);
      }
      
      Xa = a[0];
      Ya = a[1];
      Yb = a[2];
      Xc = a[3];
      Yc = a[4]; 
    }
    
    Matrix R(6,6);
    R << mob_factor[0]*(Ya-1.), 0, 0, 0, mob_factor[1]*Yb, 0,
	 0, mob_factor[0]*(Ya-1.), 0, -mob_factor[1]*Yb, 0, 0,
	 0, 0, mob_factor[0]*(Xa-1.), 0, 0, 0,
	 0, -mob_factor[1]*Yb, 0, mob_factor[2]*(Yc-4.0/3.0), 0, 0,
	 mob_factor[1]*Yb, 0, 0, 0, mob_factor[2]*(Yc-4.0/3.0), 0,
	 0, 0, 0, 0, 0, mob_factor[2]*(Xc-4.0/3.0);
         

    if(fabs(tanh_fact-1.0) > 1e-10)
    {
      R *= tanh_fact;
    }
	 
    return R;
}


Matrix c_lubrication::ResistPairLub(real r_norm, real mob_factor[3], Vector3 r_hat)
{
    real AT_cutoff = (2+0.006-1e-8);
    real WS_cutoff = (2+0.1+1e-8);
    bool inv;
    real res_factor[3] = {1.0/mob_factor[0], 1.0/mob_factor[1], 1.0/mob_factor[2]};
    Matrix R(12,12);
    
    real epsilon = r_norm-2.0;
    real tanh_fact = 1.0;
    if(epsilon < debye_cut) // epsilon < 0.1*debye_cut
    {
      epsilon = debye_cut;
      //tanh_fact = (1.0+tanh((epsilon+5*debye_cut)/2/debye_cut))*0.5;
      //epsilon = fmin(fabs(epsilon),debye_cut); //debye_cut;
      //if(epsilon < 0.1*debye_cut){epsilon = 0.1*debye_cut;}
      r_norm = epsilon+2.0; //r_norm = 2.0+debye_cut;
    }
    
    if(r_norm <= AT_cutoff)
    {
      //std::cout << "AT being used \n";
      ATResistMatrix(r_norm, mob_factor, r_hat, R);
    }
    else if(r_norm <= WS_cutoff)
    {
      inv=true;
      ResistMatrix(r_norm, res_factor, r_hat, R, inv, WS_x, mob_scalars_WS_11, mob_scalars_WS_12);
    }
    else
    {
      inv = false;
      ResistMatrix(r_norm, mob_factor, r_hat, R, inv, JO_x, mob_scalars_JO_11, mob_scalars_JO_12);
    }
    
    if(fabs(tanh_fact-1.0) > 1e-10)
    {
      R *= tanh_fact;
    }

    return R;
}

Matrix c_lubrication::ResistPairBlob(real r_norm, real mob_factor[3], Vector3 r_hat)
{
    bool inv=false;
    Matrix R(12,12);
    
    real epsilon = r_norm-2.0;
    real tanh_fact = 1.0;
    if(epsilon < debye_cut) // epsilon < 0.1*debye_cut
    {
      epsilon = debye_cut;
      //tanh_fact = (1.0+tanh((epsilon+5*debye_cut)/2/debye_cut))*0.5;
      //////////
      //epsilon = fmin(fabs(epsilon),debye_cut); //debye_cut;
      //if(epsilon < 0.1*debye_cut){epsilon = 0.1*debye_cut;}
      r_norm = epsilon+2.0; //2.0+debye_cut;
    }
    
    ResistMatrix(r_norm, mob_factor, r_hat, R, inv, MB_x, mob_scalars_MB_11, mob_scalars_MB_12);

    if(fabs(tanh_fact-1.0) > 1e-10)
    {
      R *= tanh_fact;
    }
    
    return R;
}

// TODO remove, just for testing?
void c_lubrication::ResistPairLub_py(real r_norm, real a, real eta, std::vector<real> r_hat)
{
    Vector3 r_hat_E; 
    r_hat_E << r_hat.at(0), r_hat.at(1), r_hat.at(2);
    real mob_factor[3] = {(6.0*pi_real*eta*a), (6.0*pi_real*eta*a*a), (6.0*pi_real*eta*a*a*a)};
    Matrix R = ResistPairLub(r_norm, mob_factor, r_hat_E);
        {std::cout << "[" << mob_factor[0] << " " << mob_factor[1] << " " << mob_factor[2] << "]" << std::endl;}
	{std::cout << "[" << r_hat_E[0] << " " << r_hat_E[1] << " " << r_hat_E[2] << "]" << std::endl;}
	{std::cout << r_norm << "\n"; std::cout << R << std::endl; }
}

void c_lubrication::ResistCOO(Matrix r_vectors, IMatrix n_list, real a, real eta, real cutoff, 
                             real wall_cutoff, std::vector<real> periodic_length, bool Sup_if_true, 
                             std::vector<real>& data, std::vector<int>& rows, std::vector<int>& cols)
{
  int num_bodies = r_vectors.size();
  real mob_factor[3] = {(6.0*pi_real*eta*a), (6.0*pi_real*eta*a*a), (6.0*pi_real*eta*a*a*a)};
  std::vector<real> L = periodic_length;
  int k, num_neighbors;
  Vector3 r_jk, r_hat;
  real r_norm, height;
  Matrix R_pair, R_wall, R_pair_jj, R_pair_kk, R_pair_kj, R_pair_jk;
  real R_val;
  real m_eps = 1e-12;
  
  for(int j = 0; j < num_bodies; j++)
  {
    auto r_j = r_vectors.row(j);
    height = r_j[2];
    height /= a;
    
      if(height < wall_cutoff)
      {
	if(Sup_if_true)
	{
	  R_wall = WallResistMatrix(height, mob_factor, Wall_2562_x, mob_scalars_wall_2562);
	}
	else
	{
	  R_wall = WallResistMatrixBlob(height, mob_factor, Wall_MB_x, mob_scalars_wall_MB);
	}

	for(int row = 0; row < 6; row++)
	{
	  for(int col = 0; col < 6; col++)
	  {
	    R_val = R_wall(row,col);
	    if(fabs(R_val) > m_eps)
	    {
	      data.push_back(R_val);
	      rows.push_back(row+j*6);
	      cols.push_back(col+j*6);
	    }
	  } // col  
	} // row
      }// if wall_cutoff
      
      auto neighbors = n_list.row(j);
      num_neighbors = neighbors.size();
      if(num_neighbors == 0){continue;}
      
      for(int k_ind = 0; k_ind < neighbors.size(); k_ind++)
      {
	k = neighbors[k_ind];
	
	auto r_k = r_vectors.row(k);
	for(int l = 0; l < 3; ++l)
	{
	  r_jk[l] = (r_j[l] - r_k[l]);
	  if(L.at(l) > 0)
	  {
	    r_jk[l] = r_jk[l] - int(r_jk[l] / L.at(l) + 0.5 * (int(r_jk[l]>0) - int(r_jk[l]<0))) * L.at(l);
	    r_jk[l] *= (1./a);
	  }
	}
	r_norm = r_jk.norm();
	r_hat = -r_jk/r_norm; ////// NEGATIVE SIGN
	
	if(r_norm < cutoff)
	{
	  if(Sup_if_true)
	  {
	    R_pair = ResistPairLub(r_norm, mob_factor,r_hat);
	  }
	  else
	  {
	    R_pair = ResistPairBlob(r_norm, mob_factor,r_hat);
	  }
	  R_pair_jj = R_pair.block<6,6>(0,0);
	  R_pair_kk = R_pair.block<6,6>(6,6);
	  R_pair_jk = R_pair.block<6,6>(0,6);
	  R_pair_kj = R_pair.block<6,6>(6,0);
	  
  // 	if(j == 0){std::cout << "[" << mob_factor[0] << " " << mob_factor[1] << " " << mob_factor[2] << "]" << std::endl;}
  // 	if(j == 0){std::cout << "[" << r_hat[0] << " " << r_hat[1] << " " << r_hat[2] << "]" << std::endl;}
  // 	if(j == 0){std::cout << r_norm << "\n"; std::cout << std::setprecision(5) << R_pair_jj << std::endl; }
	  
	  
	  for(int row = 0; row < 6; row++)
	  {
	    for(int col = 0; col < 6; col++)
	    {
	      // jj block
	      R_val = R_pair_jj(row,col);
	      if(fabs(R_val) > m_eps)
	      {
		data.push_back(R_val);
		rows.push_back(row+j*6);
		cols.push_back(col+j*6);
	      }
	      
	      // kk block
	      R_val = R_pair_kk(row,col);
	      if(fabs(R_val) > m_eps)
	      {
		data.push_back(R_val);
		rows.push_back(row+k*6);
		cols.push_back(col+k*6);
	      }
	      
	      // jk block
	      R_val = R_pair_jk(row,col);
	      if(fabs(R_val) > m_eps)
	      {
		data.push_back(R_val);
		rows.push_back(row+j*6);
		cols.push_back(col+k*6);
	      }
	      
	      // kj block
	      R_val = R_pair_kj(row,col);
	      if(fabs(R_val) > m_eps)
	      {
		data.push_back(R_val);
		rows.push_back(row+k*6);
		cols.push_back(col+j*6);
	      }
	      
	    } // cols
	  } // rows
	
	} // if r < cutoff
	
      }// loop over k
    
  } // loop over j
  
}

void c_lubrication::ResistCOO_wall(Matrix r_vectors, real a, real eta, real wall_cutoff, 
                    std::vector<real> periodic_length, bool Sup_if_true, std::vector<real>& data, 
                    std::vector<int>& rows, std::vector<int>& cols)
{
  int num_bodies = r_vectors.size();
  real mob_factor[3] = {(6.0*pi_real*eta*a), (6.0*pi_real*eta*a*a), (6.0*pi_real*eta*a*a*a)};
  std::vector<real> L = periodic_length;
  real r_norm, height;
  Matrix R_wall;
  real R_val;
  real m_eps = 1e-12;
  
  for(int j = 0; j < num_bodies; j++)
  {
    auto r_j = r_vectors.row(j);
    
    height = r_j[2];
    height /= a;
    
    if(height < wall_cutoff){continue;}
    
    if(Sup_if_true)
    {
      R_wall = WallResistMatrix(height, mob_factor, Wall_2562_x, mob_scalars_wall_2562);
    }
    else
    {
      R_wall = WallResistMatrixBlob(height, mob_factor, Wall_MB_x, mob_scalars_wall_MB);
    }

    for(int row = 0; row < 6; row++)
    {
      for(int col = 0; col < 6; col++)
      {
	R_val = R_wall(row,col);
	if(fabs(R_val) > m_eps)
	{
	  data.push_back(R_val);
	  rows.push_back(row+j*6);
	  cols.push_back(col+j*6);
	}
      } // col
    } // row 
  } // j loop
}

void c_lubrication::Set_R_Mats(Matrix r_vecs, IMatrix neighbors, real a, real eta, real cutoff, 
                              real cutoff_wall, std::vector<real> periodic_length, bool Sup_if_true){

  int num_particles = r_vecs.size();

  real small = 0.5*6.0*pi_real*eta*a*tolerance;

  std::vector<real> data;
  std::vector<int> rows, cols;

  ResistCOO(r_vecs, neighbors, a, eta, cutoff, cutoff_wall, periodic_length, false, data, rows, cols);

  SpMatrix R_blob_cut(6*num_particles, 6*num_particles);
  if(data.size() > 0){
    std::vector<Eigen::Triplet<real>> triplets = makeTriplets(data, rows, cols);
    R_blob_cut.setFromTriplets(triplets.begin(), triplets.end());

  } else{
    // inserts a diag
    // said to be fast creation method here: https://eigen.tuxfamily.org/dox/group__TutorialSparse.html
    for(int i = 0; i < 6*num_particles; i++){
      R_blob_cut.insert(i, i) = small;
    }
  }

  data.clear();
  rows.clear();
  cols.clear();

  ResistCOO_wall(r_vecs, a, eta, cutoff, periodic_length, false, data, rows, cols);

  SpMatrix R_blob_cut_wall(6*num_particles, 6*num_particles);
  if(data.size() > 0){
    std::vector<Eigen::Triplet<real>> triplets = makeTriplets(data, rows, cols);
    R_blob_cut_wall.setFromTriplets(triplets.begin(), triplets.end());

  } else{
    for(int i = 0; i < 6*num_particles; i++){
      R_blob_cut_wall.insert(i, i) = small;
    }
  }

  c_lubrication::R_blob = R_blob_cut + R_blob_cut_wall;


  data.clear();
  data.clear();
  data.clear();

  ResistCOO(r_vecs, neighbors, a, eta, cutoff, cutoff_wall, periodic_length, true, data, rows, cols);

  SpMatrix R_lub_cut(6*num_particles, 6*num_particles);
  if(data.size() > 0){
    std::vector<Eigen::Triplet<real>> triplets = makeTriplets(data, rows, cols);
    R_lub_cut.setFromTriplets(triplets.begin(), triplets.end());

  } else{
    for(int i = 0; i < 6*num_particles; i++){
      R_lub_cut.insert(i, i) = small;
    }
  }

  data.clear();
  rows.clear();
  cols.clear();

  ResistCOO_wall(r_vecs, a, eta, cutoff, periodic_length, true, data, rows, cols);

  SpMatrix R_lub_cut_wall(6*num_particles, 6*num_particles);
  if(data.size() > 0){
    std::vector<Eigen::Triplet<real>> triplets = makeTriplets(data, rows, cols);
    R_lub_cut_wall.setFromTriplets(triplets.begin(), triplets.end());

  } else{
    for(int i = 0; i < 6*num_particles; i++){
      R_lub_cut_wall.insert(i, i) = small;
    }
  }

  c_lubrication::R_lub = R_lub_cut + R_lub_cut_wall;

  c_lubrication::Delta_R = R_lub - R_blob;
}

// helper that assembles the COO-like format Triplets used to make eigen matrices
std::vector<Eigen::Triplet<real>> c_lubrication::makeTriplets(const std::vector<real>& data, 
                                    const std::vector<int>& rows, const std::vector<int>& cols){

  std::vector<Eigen::Triplet<real>> triplets;
  triplets.reserve(data.size());

  for(int i = 0; i < data.size(); i++){
    triplets.push_back(Eigen::Triplet<real>(rows.at(i), cols.at(i), data.at(i)));
  }

  return triplets;
}

SpMatrix c_lubrication::lub_print(){
  return Delta_R;
}

NB_MODULE(c_lubrication, m) {
  m.doc() = "c_lubrication class code";
  py::class_<c_lubrication>(m, "c_lubrication")
  .def(py::init<real>()) // c_lubrication::c_lubrication constructor
  .def("ResistCOO", &c_lubrication::ResistCOO)
  .def("ResistCOO_wall", &c_lubrication::ResistCOO_wall)
  .def("ResistPairSup_py",&c_lubrication::ResistPairLub_py)
  .def("Set_R_Mats", &c_lubrication::Set_R_Mats)
  .def("lub_print", &c_lubrication::lub_print);
}

// int main()
// {
//   c_lubrication Lub;
//   for (auto row : Lub.mob_scalars_wall_2562) {
//     for (auto el : row) {
//       std::cout << std::setprecision(16) << el << ' ';
//     }
//     std::cout << "\n";
//   }
// //   
// //   for (auto row : Lub.JO_x) {
// //       std::cout << std::setprecision(16) << row << "\n";
// //   }
//   
// //   real r_norm;
// //   real mob_factor[3] = {1.0,1.0,1.0};
// //   r_norm = 2.0084;
// //   //r_norm += 1.0;
// //   //bool inv = false;
// //   Vector3 r_hat(1.0,0.0,0.0);
// //   //Matrix R = Lub.ResistMatrix(r_norm, mob_factor, r_hat, inv, Lub.JO_x, Lub.mob_scalars_JO_11, Lub.mob_scalars_JO_12);
// //   Matrix R = Lub.ATResistMatrix(r_norm, mob_factor, r_hat);
// //   std::cout << std::setprecision(8) << R << std::endl;
//   
// 
// //   while(1){
// //       std::cin >> r_norm;
// //       int i = Lub.FindNearestIndexLower(r_norm, Lub.WS_x);
// //       std::cout << std::setprecision(16) << i << "\n";
// //   }
// //   return 0;
//   
//   real r_norm;
//   while(1){
//       std::cin >> r_norm;
//       real mob_factor[3] = {1.0,1.0,1.0};
//       Matrix R = Lub.WallResistMatrix(r_norm, mob_factor, Lub.Wall_2562_x, Lub.mob_scalars_wall_2562);
//       std::cout << std::setprecision(8) << R << "\n";
//   }
//   return 0;
//   
//   
// }
