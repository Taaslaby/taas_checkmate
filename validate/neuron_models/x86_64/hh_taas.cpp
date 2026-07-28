/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#define _pval pval
// clang-format off
#include "md1redef.h"
#include "section_fwd.hpp"
#include "nrniv_mf.h"
#include "md2redef.h"
#include "nrnconf.h"
// clang-format on
#include "neuron/cache/mechanism_range.hpp"
static constexpr auto number_of_datum_variables = 0;
static constexpr auto number_of_floating_point_variables = 16;
namespace {
template <typename T>
using _nrn_mechanism_std_vector = std::vector<T>;
using _nrn_model_sorted_token = neuron::model_sorted_token;
using _nrn_mechanism_cache_range = neuron::cache::MechanismRange<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_mechanism_cache_instance = neuron::cache::MechanismInstance<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_non_owning_id_without_container = neuron::container::non_owning_identifier_without_container;
template <typename T>
using _nrn_mechanism_field = neuron::mechanism::field<T>;
template <typename... Args>
void _nrn_mechanism_register_data_fields(Args&&... args) {
  neuron::mechanism::register_data_fields(std::forward<Args>(args)...);
}
}
 
#if !NRNGPU
#undef exp
#define exp hoc_Exp
#if NRN_ENABLE_ARCH_INDEP_EXP_POW
#undef pow
#define pow hoc_pow
#endif
#endif
 
#define nrn_init _nrn_init__hh_taas
#define _nrn_initial _nrn_initial__hh_taas
#define nrn_cur _nrn_cur__hh_taas
#define _nrn_current _nrn_current__hh_taas
#define nrn_jacob _nrn_jacob__hh_taas
#define nrn_state _nrn_state__hh_taas
#define _net_receive _net_receive__hh_taas 
#define states states__hh_taas 
 
#define _threadargscomma_ _ml, _iml, _ppvar, _thread, _globals, _nt,
#define _threadargsprotocomma_ Memb_list* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt,
#define _internalthreadargsprotocomma_ _nrn_mechanism_cache_range* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt,
#define _threadargs_ _ml, _iml, _ppvar, _thread, _globals, _nt
#define _threadargsproto_ Memb_list* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt
#define _internalthreadargsproto_ _nrn_mechanism_cache_range* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *hoc_getarg(int);
 
#define t _nt->_t
#define dt _nt->_dt
#define gnabar _ml->template fpfield<0>(_iml)
#define gnabar_columnindex 0
#define gkbar _ml->template fpfield<1>(_iml)
#define gkbar_columnindex 1
#define gl _ml->template fpfield<2>(_iml)
#define gl_columnindex 2
#define el _ml->template fpfield<3>(_iml)
#define el_columnindex 3
#define i _ml->template fpfield<4>(_iml)
#define i_columnindex 4
#define m _ml->template fpfield<5>(_iml)
#define m_columnindex 5
#define h_taas _ml->template fpfield<6>(_iml)
#define h_taas_columnindex 6
#define n _ml->template fpfield<7>(_iml)
#define n_columnindex 7
#define Dm _ml->template fpfield<8>(_iml)
#define Dm_columnindex 8
#define Dh_taas _ml->template fpfield<9>(_iml)
#define Dh_taas_columnindex 9
#define Dn _ml->template fpfield<10>(_iml)
#define Dn_columnindex 10
#define ina _ml->template fpfield<11>(_iml)
#define ina_columnindex 11
#define ik _ml->template fpfield<12>(_iml)
#define ik_columnindex 12
#define il _ml->template fpfield<13>(_iml)
#define il_columnindex 13
#define v _ml->template fpfield<14>(_iml)
#define v_columnindex 14
#define _g _ml->template fpfield<15>(_iml)
#define _g_columnindex 15
 /* Thread safe. No static _ml, _iml or _ppvar. */
 static int hoc_nrnpointerindex =  -1;
 static _nrn_mechanism_std_vector<Datum> _extcall_thread;
 static Prop* _extcall_prop;
 /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
 static _nrn_non_owning_id_without_container _prop_id{};
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_alpha_n(void);
 static void _hoc_alpha_h(void);
 static void _hoc_alpha_m(void);
 static void _hoc_beta_n(void);
 static void _hoc_beta_h(void);
 static void _hoc_beta_m(void);
 static void _hoc_vtrap(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mechtype);
#endif
 static void _hoc_setdata();
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 {"setdata_hh_taas", _hoc_setdata},
 {"alpha_n_hh_taas", _hoc_alpha_n},
 {"alpha_h_hh_taas", _hoc_alpha_h},
 {"alpha_m_hh_taas", _hoc_alpha_m},
 {"beta_n_hh_taas", _hoc_beta_n},
 {"beta_h_hh_taas", _hoc_beta_h},
 {"beta_m_hh_taas", _hoc_beta_m},
 {"vtrap_hh_taas", _hoc_vtrap},
 {0, 0}
};
 
/* Direct Python call wrappers to density mechanism functions.*/
 static double _npy_alpha_n(Prop*);
 static double _npy_alpha_h(Prop*);
 static double _npy_alpha_m(Prop*);
 static double _npy_beta_n(Prop*);
 static double _npy_beta_h(Prop*);
 static double _npy_beta_m(Prop*);
 static double _npy_vtrap(Prop*);
 
static NPyDirectMechFunc npy_direct_func_proc[] = {
 {"alpha_n", _npy_alpha_n},
 {"alpha_h", _npy_alpha_h},
 {"alpha_m", _npy_alpha_m},
 {"beta_n", _npy_beta_n},
 {"beta_h", _npy_beta_h},
 {"beta_m", _npy_beta_m},
 {"vtrap", _npy_vtrap},
 {0, 0}
};
#define alpha_n alpha_n_hh_taas
#define alpha_h alpha_h_hh_taas
#define alpha_m alpha_m_hh_taas
#define beta_n beta_n_hh_taas
#define beta_h beta_h_hh_taas
#define beta_m beta_m_hh_taas
#define vtrap vtrap_hh_taas
 extern double alpha_n( _internalthreadargsprotocomma_ double );
 extern double alpha_h( _internalthreadargsprotocomma_ double );
 extern double alpha_m( _internalthreadargsprotocomma_ double );
 extern double beta_n( _internalthreadargsprotocomma_ double );
 extern double beta_h( _internalthreadargsprotocomma_ double );
 extern double beta_m( _internalthreadargsprotocomma_ double );
 extern double vtrap( _internalthreadargsprotocomma_ double , double );
 /* declare global and static user variables */
 #define gind 0
 #define _gth 0
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"gnabar_hh_taas", "S/cm2"},
 {"gkbar_hh_taas", "S/cm2"},
 {"gl_hh_taas", "S/cm2"},
 {"el_hh_taas", "mV"},
 {"i_hh_taas", "mA/cm2"},
 {0, 0}
};
 static double delta_t = 0.01;
 static double h_taas0 = 0;
 static double m0 = 0;
 static double n0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {0, 0}
};
 static DoubVec hoc_vdoub[] = {
 {0, 0, 0}
};
 static double _sav_indep;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 _prop_id = _nrn_get_prop_id(_prop);
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 static void nrn_alloc(Prop*);
static void nrn_init(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void nrn_state(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 static void nrn_cur(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void nrn_jacob(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(Prop*, int, neuron::container::data_handle<double>*, neuron::container::data_handle<double>*, double*, int);
static void _ode_spec(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void _ode_matsol(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 
#define _cvode_ieq _ppvar[0].literal_value<int>()
 static void _ode_matsol_instance1(_internalthreadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"hh_taas",
 "gnabar_hh_taas",
 "gkbar_hh_taas",
 "gl_hh_taas",
 "el_hh_taas",
 0,
 "i_hh_taas",
 0,
 "m_hh_taas",
 "h_taas_hh_taas",
 "n_hh_taas",
 0,
 0};
 
 /* Used by NrnProperty */
 static _nrn_mechanism_std_vector<double> _parm_default{
     0.12, /* gnabar */
     0.036, /* gkbar */
     0.0003, /* gl */
     -54.3, /* el */
 }; 
 
 
extern Prop* need_memb(Symbol*);
static void nrn_alloc(Prop* _prop) {
  Prop *prop_ion{};
  Datum *_ppvar{};
   _ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
    _nrn_mechanism_access_dparam(_prop) = _ppvar;
     _nrn_mechanism_cache_instance _ml_real{_prop};
    auto* const _ml = &_ml_real;
    size_t const _iml{};
    assert(_nrn_mechanism_get_num_vars(_prop) == 16);
 	/*initialize range parameters*/
 	gnabar = _parm_default[0]; /* 0.12 */
 	gkbar = _parm_default[1]; /* 0.036 */
 	gl = _parm_default[2]; /* 0.0003 */
 	el = _parm_default[3]; /* -54.3 */
 	 assert(_nrn_mechanism_get_num_vars(_prop) == 16);
 	_nrn_mechanism_access_dparam(_prop) = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _hh_taas_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
 hoc_register_parm_default(_mechtype, &_parm_default);
         hoc_register_npy_direct(_mechtype, npy_direct_func_proc);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
   _nrn_mechanism_register_data_fields(_mechtype,
                                       _nrn_mechanism_field<double>{"gnabar"} /* 0 */,
                                       _nrn_mechanism_field<double>{"gkbar"} /* 1 */,
                                       _nrn_mechanism_field<double>{"gl"} /* 2 */,
                                       _nrn_mechanism_field<double>{"el"} /* 3 */,
                                       _nrn_mechanism_field<double>{"i"} /* 4 */,
                                       _nrn_mechanism_field<double>{"m"} /* 5 */,
                                       _nrn_mechanism_field<double>{"h_taas"} /* 6 */,
                                       _nrn_mechanism_field<double>{"n"} /* 7 */,
                                       _nrn_mechanism_field<double>{"Dm"} /* 8 */,
                                       _nrn_mechanism_field<double>{"Dh_taas"} /* 9 */,
                                       _nrn_mechanism_field<double>{"Dn"} /* 10 */,
                                       _nrn_mechanism_field<double>{"ina"} /* 11 */,
                                       _nrn_mechanism_field<double>{"ik"} /* 12 */,
                                       _nrn_mechanism_field<double>{"il"} /* 13 */,
                                       _nrn_mechanism_field<double>{"v"} /* 14 */,
                                       _nrn_mechanism_field<double>{"_g"} /* 15 */,
                                       _nrn_mechanism_field<int>{"_cvode_ieq", "cvodeieq"} /* 0 */);
  hoc_register_prop_size(_mechtype, 16, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 
    hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 hh_taas /home/taaslab/Проекты/asm_engines/taas_checkmate/validate/neuron_models/hh_taas.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static const char *modelname = "Hodgkin-Huxley (taas_checkmate version)";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_internalthreadargsproto_);
/*static int _ode_matsol1(_internalthreadargsproto_);*/
 static neuron::container::field_index _slist1[3], _dlist1[3];
 static int states(_internalthreadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (_internalthreadargsproto_) {int _reset = 0; {
   Dm = ( alpha_m ( _threadargscomma_ v ) - ( alpha_m ( _threadargscomma_ v ) + beta_m ( _threadargscomma_ v ) ) * m ) ;
   Dh_taas = ( alpha_h ( _threadargscomma_ v ) - ( alpha_h ( _threadargscomma_ v ) + beta_h ( _threadargscomma_ v ) ) * h_taas ) ;
   Dn = ( alpha_n ( _threadargscomma_ v ) - ( alpha_n ( _threadargscomma_ v ) + beta_n ( _threadargscomma_ v ) ) * n ) ;
   }
 return _reset;
}
 static int _ode_matsol1 (_internalthreadargsproto_) {
 Dm = Dm  / (1. - dt*( ( ( - ( ( alpha_m ( _threadargscomma_ v ) + beta_m ( _threadargscomma_ v ) ) )*( 1.0 ) ) ) )) ;
 Dh_taas = Dh_taas  / (1. - dt*( ( ( - ( ( alpha_h ( _threadargscomma_ v ) + beta_h ( _threadargscomma_ v ) ) )*( 1.0 ) ) ) )) ;
 Dn = Dn  / (1. - dt*( ( ( - ( ( alpha_n ( _threadargscomma_ v ) + beta_n ( _threadargscomma_ v ) ) )*( 1.0 ) ) ) )) ;
  return 0;
}
 /*END CVODE*/
 static int states (_internalthreadargsproto_) { {
    m = m + (1. - exp(dt*(( ( - ( ( alpha_m ( _threadargscomma_ v ) + beta_m ( _threadargscomma_ v ) ) )*( 1.0 ) ) ))))*(- ( ( alpha_m ( _threadargscomma_ v ) ) ) / ( ( ( - ( ( alpha_m ( _threadargscomma_ v ) + beta_m ( _threadargscomma_ v ) ) )*( 1.0 ) ) ) ) - m) ;
    h_taas = h_taas + (1. - exp(dt*(( ( - ( ( alpha_h ( _threadargscomma_ v ) + beta_h ( _threadargscomma_ v ) ) )*( 1.0 ) ) ))))*(- ( ( alpha_h ( _threadargscomma_ v ) ) ) / ( ( ( - ( ( alpha_h ( _threadargscomma_ v ) + beta_h ( _threadargscomma_ v ) ) )*( 1.0 ) ) ) ) - h_taas) ;
    n = n + (1. - exp(dt*(( ( - ( ( alpha_n ( _threadargscomma_ v ) + beta_n ( _threadargscomma_ v ) ) )*( 1.0 ) ) ))))*(- ( ( alpha_n ( _threadargscomma_ v ) ) ) / ( ( ( - ( ( alpha_n ( _threadargscomma_ v ) + beta_n ( _threadargscomma_ v ) ) )*( 1.0 ) ) ) ) - n) ;
   }
  return 0;
}
 
double alpha_m ( _internalthreadargsprotocomma_ double _lv ) {
   double _lalpha_m;
 _lalpha_m = 0.1 * vtrap ( _threadargscomma_ - ( _lv + 40.0 ) , 10.0 ) ;
   
return _lalpha_m;
 }
 
static void _hoc_alpha_m(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  alpha_m ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_alpha_m(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  alpha_m ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
double beta_m ( _internalthreadargsprotocomma_ double _lv ) {
   double _lbeta_m;
 _lbeta_m = 4.0 * exp ( - ( _lv + 65.0 ) / 18.0 ) ;
   
return _lbeta_m;
 }
 
static void _hoc_beta_m(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  beta_m ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_beta_m(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  beta_m ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
double alpha_h ( _internalthreadargsprotocomma_ double _lv ) {
   double _lalpha_h;
 _lalpha_h = 0.07 * exp ( - ( _lv + 65.0 ) / 20.0 ) ;
   
return _lalpha_h;
 }
 
static void _hoc_alpha_h(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  alpha_h ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_alpha_h(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  alpha_h ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
double beta_h ( _internalthreadargsprotocomma_ double _lv ) {
   double _lbeta_h;
 _lbeta_h = 1.0 / ( 1.0 + exp ( - ( _lv + 35.0 ) / 10.0 ) ) ;
   
return _lbeta_h;
 }
 
static void _hoc_beta_h(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  beta_h ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_beta_h(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  beta_h ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
double alpha_n ( _internalthreadargsprotocomma_ double _lv ) {
   double _lalpha_n;
 _lalpha_n = 0.01 * vtrap ( _threadargscomma_ - ( _lv + 55.0 ) , 10.0 ) ;
   
return _lalpha_n;
 }
 
static void _hoc_alpha_n(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  alpha_n ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_alpha_n(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  alpha_n ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
double beta_n ( _internalthreadargsprotocomma_ double _lv ) {
   double _lbeta_n;
 _lbeta_n = 0.125 * exp ( - ( _lv + 65.0 ) / 80.0 ) ;
   
return _lbeta_n;
 }
 
static void _hoc_beta_n(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  beta_n ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_beta_n(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  beta_n ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
double vtrap ( _internalthreadargsprotocomma_ double _lx , double _ly ) {
   double _lvtrap;
 if ( fabs ( _lx / _ly ) < 1e-6 ) {
     _lvtrap = _ly * ( 1.0 - _lx / _ly / 2.0 ) ;
     }
   else {
     _lvtrap = _lx / ( exp ( _lx / _ly ) - 1.0 ) ;
     }
   
return _lvtrap;
 }
 
static void _hoc_vtrap(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  vtrap ( _threadargscomma_ *getarg(1) , *getarg(2) );
 hoc_retpushx(_r);
}
 
static double _npy_vtrap(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 _r =  vtrap ( _threadargscomma_ *getarg(1) , *getarg(2) );
 return(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
   Datum* _ppvar;
   size_t _iml;   _nrn_mechanism_cache_range* _ml;   Node* _nd{};
  double _v{};
  int _cntml;
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
  _ml = &_lmr;
  _cntml = _ml_arg->_nodecount;
  Datum *_thread{_ml_arg->_thread};
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _ppvar = _ml_arg->_pdata[_iml];
    _nd = _ml_arg->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_threadargs_);
 }}
 
static void _ode_map(Prop* _prop, int _ieq, neuron::container::data_handle<double>* _pv, neuron::container::data_handle<double>* _pvdot, double* _atol, int _type) { 
  Datum* _ppvar;
  _ppvar = _nrn_mechanism_access_dparam(_prop);
  _cvode_ieq = _ieq;
  for (int _i=0; _i < 3; ++_i) {
    _pv[_i] = _nrn_mechanism_get_param_handle(_prop, _slist1[_i]);
    _pvdot[_i] = _nrn_mechanism_get_param_handle(_prop, _dlist1[_i]);
    _cvode_abstol(_atollist, _atol, _i);
  }
 }
 
static void _ode_matsol_instance1(_internalthreadargsproto_) {
 _ode_matsol1 (_threadargs_);
 }
 
static void _ode_matsol(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
   Datum* _ppvar;
   size_t _iml;   _nrn_mechanism_cache_range* _ml;   Node* _nd{};
  double _v{};
  int _cntml;
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
  _ml = &_lmr;
  _cntml = _ml_arg->_nodecount;
  Datum *_thread{_ml_arg->_thread};
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _ppvar = _ml_arg->_pdata[_iml];
    _nd = _ml_arg->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(_internalthreadargsproto_) {
  int _i; double _save;{
  h_taas = h_taas0;
  m = m0;
  n = n0;
 {
   m = alpha_m ( _threadargscomma_ v ) / ( alpha_m ( _threadargscomma_ v ) + beta_m ( _threadargscomma_ v ) ) ;
   h_taas = alpha_h ( _threadargscomma_ v ) / ( alpha_h ( _threadargscomma_ v ) + beta_h ( _threadargscomma_ v ) ) ;
   n = alpha_n ( _threadargscomma_ v ) / ( alpha_n ( _threadargscomma_ v ) + beta_n ( _threadargscomma_ v ) ) ;
   }
 
}
}

static void nrn_init(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type){
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
   _v = _vec_v[_ni[_iml]];
 v = _v;
 initmodel(_threadargs_);
}
}

static double _nrn_current(_internalthreadargsprotocomma_ double _v) {
double _current=0.; v=_v;
{ {
   ina = gnabar * m * m * m * h_taas * ( v - 50.0 ) ;
   ik = gkbar * n * n * n * n * ( v + 77.0 ) ;
   il = gl * ( v - el ) ;
   i = ina + ik + il ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto const _vec_rhs = _nt->node_rhs_storage();
auto const _vec_sav_rhs = _nt->node_sav_rhs_storage();
auto const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
   _v = _vec_v[_ni[_iml]];
 auto const _g_local = _nrn_current(_threadargscomma_ _v + .001);
 	{ _rhs = _nrn_current(_threadargscomma_ _v);
 	}
 _g = (_g_local - _rhs)/.001;
	 _vec_rhs[_ni[_iml]] -= _rhs;
 
}
 
}

static void nrn_jacob(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto const _vec_d = _nt->node_d_storage();
auto const _vec_sav_d = _nt->node_sav_d_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
  _vec_d[_ni[_iml]] += _g;
 
}
 
}

static void nrn_state(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni;
_ni = _ml_arg->_nodeindices;
size_t _cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (size_t _iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
 _nd = _ml_arg->_nodelist[_iml];
   _v = _vec_v[_ni[_iml]];
 v=_v;
{
 {   states(_threadargs_);
  }}}

}

static void terminal(){}

static void _initlists(){
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = {m_columnindex, 0};  _dlist1[0] = {Dm_columnindex, 0};
 _slist1[1] = {h_taas_columnindex, 0};  _dlist1[1] = {Dh_taas_columnindex, 0};
 _slist1[2] = {n_columnindex, 0};  _dlist1[2] = {Dn_columnindex, 0};
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/home/taaslab/Проекты/asm_engines/taas_checkmate/validate/neuron_models/hh_taas.mod";
    const char* nmodl_file_text = 
  "TITLE Hodgkin-Huxley (taas_checkmate version)\n"
  ": Standard HH sodium + potassium + leak\n"
  ": NEURON 9.0 compatible\n"
  "\n"
  "NEURON {\n"
  "    SUFFIX hh_taas\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    RANGE gnabar, gkbar, gl, el\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "    (mA) = (milliamp)\n"
  "    (mV) = (millivolt)\n"
  "    (S) = (siemens)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    gnabar = 0.12 (S/cm2)\n"
  "    gkbar = 0.036 (S/cm2)\n"
  "    gl = 0.0003 (S/cm2)\n"
  "    el = -54.3 (mV)\n"
  "}\n"
  "\n"
  "STATE { m h_taas n }\n"
  "\n"
  "ASSIGNED {\n"
  "    i (mA/cm2)\n"
  "    ina (mA/cm2)\n"
  "    ik (mA/cm2)\n"
  "    il (mA/cm2)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    m = alpha_m(v)/(alpha_m(v) + beta_m(v))\n"
  "    h_taas = alpha_h(v)/(alpha_h(v) + beta_h(v))\n"
  "    n = alpha_n(v)/(alpha_n(v) + beta_n(v))\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE states METHOD cnexp\n"
  "    ina = gnabar * m*m*m * h_taas * (v - 50)\n"
  "    ik = gkbar * n*n*n*n * (v + 77)\n"
  "    il = gl * (v - el)\n"
  "    i = ina + ik + il\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "    m' = (alpha_m(v) - (alpha_m(v)+beta_m(v))*m)\n"
  "    h_taas' = (alpha_h(v) - (alpha_h(v)+beta_h(v))*h_taas)\n"
  "    n' = (alpha_n(v) - (alpha_n(v)+beta_n(v))*n)\n"
  "}\n"
  "\n"
  "FUNCTION alpha_m(v) {\n"
  "    alpha_m = 0.1*vtrap(-(v+40), 10)\n"
  "}\n"
  "\n"
  "FUNCTION beta_m(v) {\n"
  "    beta_m = 4*exp(-(v+65)/18)\n"
  "}\n"
  "\n"
  "FUNCTION alpha_h(v) {\n"
  "    alpha_h = 0.07*exp(-(v+65)/20)\n"
  "}\n"
  "\n"
  "FUNCTION beta_h(v) {\n"
  "    beta_h = 1/(1+exp(-(v+35)/10))\n"
  "}\n"
  "\n"
  "FUNCTION alpha_n(v) {\n"
  "    alpha_n = 0.01*vtrap(-(v+55), 10)\n"
  "}\n"
  "\n"
  "FUNCTION beta_n(v) {\n"
  "    beta_n = 0.125*exp(-(v+65)/80)\n"
  "}\n"
  "\n"
  "FUNCTION vtrap(x, y) {\n"
  "    if (fabs(x/y) < 1e-6) {\n"
  "        vtrap = y*(1 - x/y/2)\n"
  "    } else {\n"
  "        vtrap = x/(exp(x/y) - 1)\n"
  "    }\n"
  "}\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
