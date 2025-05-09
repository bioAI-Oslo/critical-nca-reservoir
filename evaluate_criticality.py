import os
import numpy as np
from critical_nca import CriticalNCA
import powerlaw
from sklearn.linear_model import LinearRegression
import utils
# import matplotlib.image
import matplotlib.pyplot as plt
import time
from PIL import Image
from skimage import measure

plt.rcParams.update({'font.size': 14})

width = 1000
timesteps = 1000

def KSdist(theoretical_pdf, empirical_pdf):
  return np.max(np.abs(np.cumsum(theoretical_pdf) - np.cumsum(empirical_pdf)))

def getdict_cluster_size(arr1d):
  cluster_dict = {}
  current_number = None
  for a in arr1d:
    if current_number == a:
      cluster_dict[a][-1] = cluster_dict[a][-1]+1
    else:
      current_number = a
      if a in cluster_dict:
        cluster_dict[a].append(1)
      else:
        cluster_dict[a] = [1]
  return cluster_dict

def getarray_avalanche_size(x, value):
  list_avalance_size = []
  if value in x:
    x0size, x1size = x.shape
    for i in range(x0size):
      if value in x[i,:]:
        list_avalance_size.extend(getdict_cluster_size(x[i,:])[value])
  return np.array(list_avalance_size)

def getarray_avalanche_duration(x, value):
  list_avalance_duration = []
  if value in x:
    x0size, x1size = x.shape
    for i in range(x1size):
      if value in x[:,i]:
        list_avalance_duration.extend(getdict_cluster_size(x[:,i])[value])
  return np.array(list_avalance_duration)

def goodness_of_fit(fit, data, xmin=1, gen_data=1000, data_samples_lb=10000):
  theoretical_distribution = powerlaw.Power_Law(xmin=xmin,\
                                                parameters=[fit.power_law.alpha],\
                                                discrete=True)
  simulated_ksdist_list = []
  data_samples = max(len(data), data_samples_lb)
  print("GoF data_samples", data_samples)
  for _ in range(gen_data):
    simulated_data=theoretical_distribution.generate_random(data_samples)
    simulated_ksdist = powerlaw.power_law_ks_distance(simulated_data,\
                                                      fit.power_law.alpha,\
                                                      xmin=xmin, xmax=1000, discrete=True)
    simulated_ksdist_list.append(simulated_ksdist)

  return sum(np.array(simulated_ksdist_list) > fit.power_law.D) / gen_data

def norm_coef(coef):
  return -np.mean(coef)

def norm_linscore(linscore):
  return np.mean(linscore)

# Normalize values from 0 to inf to be from 10 to 0
def norm_ksdist(ksdist, smooth=1):
  return np.exp(-smooth * (0.9*min(np.mean(ksdist[:3]), np.mean(ksdist[3:]))+0.1*np.mean(ksdist)))



# Normalize values from -inf to inf to be from 0 to 1
def norm_R(R, smooth=0.01):
  return 1. / (1.+np.exp(-smooth * (0.9*max(np.mean(R[:3]), np.mean(R[3:]))+0.1*np.mean(R))))

def normalize_avalanche_pdf_size(mask_avalanche_s_0_bc, mask_avalanche_d_0_bc,\
                                 mask_avalanche_t_0_bc, mask_avalanche_s_1_bc,\
                                 mask_avalanche_d_1_bc, mask_avalanche_t_1_bc):
  norm_avalanche_pdf_size_s_0 = sum(mask_avalanche_s_0_bc)/width
  norm_avalanche_pdf_size_d_0 = sum(mask_avalanche_d_0_bc)/timesteps
  norm_avalanche_pdf_size_t_0 = sum(mask_avalanche_t_0_bc)/(timesteps*width)
  norm_avalanche_pdf_size_s_1 = sum(mask_avalanche_s_1_bc)/width
  norm_avalanche_pdf_size_d_1 = sum(mask_avalanche_d_1_bc)/timesteps
  norm_avalanche_pdf_size_t_1 = sum(mask_avalanche_t_1_bc)/(timesteps*width)

  mean_avalanche_pdf_size = np.mean([norm_avalanche_pdf_size_s_0,\
                                    norm_avalanche_pdf_size_d_0,\
                                    norm_avalanche_pdf_size_t_0,\
                                    norm_avalanche_pdf_size_s_1,\
                                    norm_avalanche_pdf_size_d_1,\
                                    norm_avalanche_pdf_size_t_1])

  max_avalanche_pdf_size = max(np.mean([norm_avalanche_pdf_size_s_0,\
                                    norm_avalanche_pdf_size_d_0,\
                                    norm_avalanche_pdf_size_t_0]),\
                                    np.mean([norm_avalanche_pdf_size_s_1,\
                                    norm_avalanche_pdf_size_d_1,\
                                    norm_avalanche_pdf_size_t_1]))


  return np.tanh(5*(0.9*max_avalanche_pdf_size+0.1*mean_avalanche_pdf_size))

def normalize_avalanche_pdf_size_2(mask_avalanche_s_0_bc, mask_avalanche_d_0_bc,\
                                 mask_avalanche_s_1_bc, mask_avalanche_d_1_bc):
  norm_avalanche_pdf_size_s_0 = sum(mask_avalanche_s_0_bc)/width
  norm_avalanche_pdf_size_d_0 = sum(mask_avalanche_d_0_bc)/timesteps
  norm_avalanche_pdf_size_s_1 = sum(mask_avalanche_s_1_bc)/width
  norm_avalanche_pdf_size_d_1 = sum(mask_avalanche_d_1_bc)/timesteps

  mean_avalanche_pdf_size = np.mean([norm_avalanche_pdf_size_s_0,\
                                    norm_avalanche_pdf_size_d_0,\
                                    norm_avalanche_pdf_size_s_1,\
                                    norm_avalanche_pdf_size_d_1])
  max_avalanche_pdf_size = np.max([norm_avalanche_pdf_size_s_0,\
                                   norm_avalanche_pdf_size_d_0,\
                                   norm_avalanche_pdf_size_s_1,\
                                   norm_avalanche_pdf_size_d_1])

  return np.tanh(5*(0.9*max_avalanche_pdf_size+0.1*mean_avalanche_pdf_size))

def sigmoid(x, smooth=0.01):
  return 1. / (1. + np.exp(-x*smooth))

def norm_comparison_ratio(R_list):
  return sigmoid(0.9*max(np.mean(R_list[:3]), np.mean(R_list[3:])) + 0.1*np.mean(R_list))

def calculate_comparison_ratio(data):
  fit = powerlaw.Fit(data, xmin =1, discrete= True)
  R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
  R = R_exp if p_exp < 0.1 else 0

  return R



def evaluate_result(ca_result, filename=None):
  avalanche_s_0, avalanche_d_0, avalanche_t_0 = getarray_avalanche_size_duration_total(ca_result, 0, save_plot=True)
  avalanche_s_1, avalanche_d_1, avalanche_t_1 = getarray_avalanche_size_duration_total(ca_result, 1, save_plot=True)

  avalanche_s_0_bc = np.bincount(avalanche_s_0)[1:] if len(avalanche_s_0) > 5 else np.array([])
  avalanche_d_0_bc = np.bincount(avalanche_d_0)[1:] if len(avalanche_d_0) > 5 else np.array([])
  avalanche_t_0_bc = np.bincount(avalanche_t_0)[1:] if len(avalanche_t_0) > 5 else np.array([])

  avalanche_s_1_bc = np.bincount(avalanche_s_1)[1:] if len(avalanche_s_1) > 5 else np.array([])
  avalanche_d_1_bc = np.bincount(avalanche_d_1)[1:] if len(avalanche_d_1) > 5 else np.array([])
  avalanche_t_1_bc = np.bincount(avalanche_t_1)[1:] if len(avalanche_t_1) > 5 else np.array([])


  avalanche_s_0_bc = avalanche_s_0_bc/sum(avalanche_s_0_bc)
  avalanche_d_0_bc = avalanche_d_0_bc/sum(avalanche_d_0_bc)
  avalanche_t_0_bc = avalanche_t_0_bc/sum(avalanche_t_0_bc)
  avalanche_s_1_bc = avalanche_s_1_bc/sum(avalanche_s_1_bc)
  avalanche_d_1_bc = avalanche_d_1_bc/sum(avalanche_d_1_bc)
  avalanche_t_1_bc = avalanche_t_1_bc/sum(avalanche_t_1_bc)

  mask_avalanche_s_0_bc = avalanche_s_0_bc > 0
  mask_avalanche_d_0_bc = avalanche_d_0_bc > 0
  mask_avalanche_t_0_bc = avalanche_t_0_bc > 0
  mask_avalanche_s_1_bc = avalanche_s_1_bc > 0
  mask_avalanche_d_1_bc = avalanche_d_1_bc > 0
  mask_avalanche_t_1_bc = avalanche_t_1_bc > 0

  log_avalanche_s_0_bc = np.log10(avalanche_s_0_bc)
  log_avalanche_d_0_bc = np.log10(avalanche_d_0_bc)
  log_avalanche_t_0_bc = np.log10(avalanche_t_0_bc)
  log_avalanche_s_1_bc = np.log10(avalanche_s_1_bc)
  log_avalanche_d_1_bc = np.log10(avalanche_d_1_bc)
  log_avalanche_t_1_bc = np.log10(avalanche_t_1_bc)

  log_avalanche_s_0_bc = np.where(mask_avalanche_s_0_bc, log_avalanche_s_0_bc, 0)
  log_avalanche_d_0_bc = np.where(mask_avalanche_d_0_bc, log_avalanche_d_0_bc, 0)
  log_avalanche_t_0_bc = np.where(mask_avalanche_t_0_bc, log_avalanche_t_0_bc, 0)
  log_avalanche_s_1_bc = np.where(mask_avalanche_s_1_bc, log_avalanche_s_1_bc, 0)
  log_avalanche_d_1_bc = np.where(mask_avalanche_d_1_bc, log_avalanche_d_1_bc, 0)
  log_avalanche_t_1_bc = np.where(mask_avalanche_t_1_bc, log_avalanche_t_1_bc, 0)

  fitness = 0
  norm_avalanche_pdf_size = 0
  norm_linscore_res = 0
  norm_ksdist_res = 0
  norm_coef_res = 0
  norm_unique_states = 0
  norm_R_res = 0

  if sum(mask_avalanche_s_0_bc[:10]) > 5 and sum(mask_avalanche_d_0_bc[:10]) > 5 and\
    sum(mask_avalanche_t_0_bc[:10]) > 5 and sum(mask_avalanche_s_1_bc[:10]) > 5 and\
    sum(mask_avalanche_d_1_bc[:10]) > 5 and sum(mask_avalanche_t_1_bc[:10]) > 5:

    # Fit PDF using least square error
    fit_avalanche_s_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_0_bc)+1)[mask_avalanche_s_0_bc]).reshape(-1,1), log_avalanche_s_0_bc[mask_avalanche_s_0_bc])
    fit_avalanche_d_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_0_bc)+1)[mask_avalanche_d_0_bc]).reshape(-1,1), log_avalanche_d_0_bc[mask_avalanche_d_0_bc])
    fit_avalanche_t_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_t_0_bc)+1)[mask_avalanche_t_0_bc]).reshape(-1,1), log_avalanche_t_0_bc[mask_avalanche_t_0_bc])
    fit_avalanche_s_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_1_bc)+1)[mask_avalanche_s_1_bc]).reshape(-1,1), log_avalanche_s_1_bc[mask_avalanche_s_1_bc])
    fit_avalanche_d_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_1_bc)+1)[mask_avalanche_d_1_bc]).reshape(-1,1), log_avalanche_d_1_bc[mask_avalanche_d_1_bc])
    fit_avalanche_t_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_t_1_bc)+1)[mask_avalanche_t_1_bc]).reshape(-1,1), log_avalanche_t_1_bc[mask_avalanche_t_1_bc])

    linscore_list = []
    linscore_list.append(fit_avalanche_s_0_bc.score(np.log10(np.arange(1,len(avalanche_s_0_bc)+1)[mask_avalanche_s_0_bc]).reshape(-1,1), log_avalanche_s_0_bc[mask_avalanche_s_0_bc]))
    linscore_list.append(fit_avalanche_d_0_bc.score(np.log10(np.arange(1,len(avalanche_d_0_bc)+1)[mask_avalanche_d_0_bc]).reshape(-1,1), log_avalanche_d_0_bc[mask_avalanche_d_0_bc]))
    linscore_list.append(fit_avalanche_t_0_bc.score(np.log10(np.arange(1,len(avalanche_t_0_bc)+1)[mask_avalanche_t_0_bc]).reshape(-1,1), log_avalanche_t_0_bc[mask_avalanche_t_0_bc]))
    linscore_list.append(fit_avalanche_s_1_bc.score(np.log10(np.arange(1,len(avalanche_s_1_bc)+1)[mask_avalanche_s_1_bc]).reshape(-1,1), log_avalanche_s_1_bc[mask_avalanche_s_1_bc]))
    linscore_list.append(fit_avalanche_d_1_bc.score(np.log10(np.arange(1,len(avalanche_d_1_bc)+1)[mask_avalanche_d_1_bc]).reshape(-1,1), log_avalanche_d_1_bc[mask_avalanche_d_1_bc]))
    linscore_list.append(fit_avalanche_t_1_bc.score(np.log10(np.arange(1,len(avalanche_t_1_bc)+1)[mask_avalanche_t_1_bc]).reshape(-1,1), log_avalanche_t_1_bc[mask_avalanche_t_1_bc]))

    # Fit PDF using least square error
    fit_avalanche_s_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_0_bc)+1)[mask_avalanche_s_0_bc]).reshape(-1,1), log_avalanche_s_0_bc[mask_avalanche_s_0_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_s_0_bc))[mask_avalanche_s_0_bc]])
    fit_avalanche_d_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_0_bc)+1)[mask_avalanche_d_0_bc]).reshape(-1,1), log_avalanche_d_0_bc[mask_avalanche_d_0_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_d_0_bc))[mask_avalanche_d_0_bc]])
    fit_avalanche_t_0_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_t_0_bc)+1)[mask_avalanche_t_0_bc]).reshape(-1,1), log_avalanche_t_0_bc[mask_avalanche_t_0_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_t_0_bc))[mask_avalanche_t_0_bc]])
    fit_avalanche_s_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_s_1_bc)+1)[mask_avalanche_s_1_bc]).reshape(-1,1), log_avalanche_s_1_bc[mask_avalanche_s_1_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_s_1_bc))[mask_avalanche_s_1_bc]])
    fit_avalanche_d_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_d_1_bc)+1)[mask_avalanche_d_1_bc]).reshape(-1,1), log_avalanche_d_1_bc[mask_avalanche_d_1_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_d_1_bc))[mask_avalanche_d_1_bc]])
    fit_avalanche_t_1_bc = LinearRegression().fit(np.log10(np.arange(1,len(avalanche_t_1_bc)+1)[mask_avalanche_t_1_bc]).reshape(-1,1), log_avalanche_t_1_bc[mask_avalanche_t_1_bc], sample_weight=[1 if idx < 10 else 0 for idx in np.arange(len(avalanche_t_1_bc))[mask_avalanche_t_1_bc]])

    theor_avalanche_s_0_bc = np.power(10,fit_avalanche_s_0_bc.predict(np.log10(np.arange(1,len(avalanche_s_0_bc)+1).reshape(-1,1))))
    theor_avalanche_d_0_bc = np.power(10,fit_avalanche_d_0_bc.predict(np.log10(np.arange(1,len(avalanche_d_0_bc)+1).reshape(-1,1))))
    theor_avalanche_t_0_bc = np.power(10,fit_avalanche_t_0_bc.predict(np.log10(np.arange(1,len(avalanche_t_0_bc)+1).reshape(-1,1))))
    theor_avalanche_s_1_bc = np.power(10,fit_avalanche_s_1_bc.predict(np.log10(np.arange(1,len(avalanche_s_1_bc)+1).reshape(-1,1))))
    theor_avalanche_d_1_bc = np.power(10,fit_avalanche_d_1_bc.predict(np.log10(np.arange(1,len(avalanche_d_1_bc)+1).reshape(-1,1))))
    theor_avalanche_t_1_bc = np.power(10,fit_avalanche_t_1_bc.predict(np.log10(np.arange(1,len(avalanche_t_1_bc)+1).reshape(-1,1))))

    ksdist_list = []
    ksdist_list.append(KSdist(theor_avalanche_s_0_bc, avalanche_s_0_bc))
    ksdist_list.append(KSdist(theor_avalanche_d_0_bc, avalanche_d_0_bc))
    ksdist_list.append(KSdist(theor_avalanche_t_0_bc, avalanche_t_0_bc))
    ksdist_list.append(KSdist(theor_avalanche_s_1_bc, avalanche_s_1_bc))
    ksdist_list.append(KSdist(theor_avalanche_d_1_bc, avalanche_d_1_bc))
    ksdist_list.append(KSdist(theor_avalanche_t_1_bc, avalanche_t_1_bc))

    coef_list = []
    coef_list.append(fit_avalanche_s_0_bc.coef_[0])
    coef_list.append(fit_avalanche_d_0_bc.coef_[0])
    coef_list.append(fit_avalanche_t_0_bc.coef_[0])
    coef_list.append(fit_avalanche_s_1_bc.coef_[0])
    coef_list.append(fit_avalanche_d_1_bc.coef_[0])
    coef_list.append(fit_avalanche_t_1_bc.coef_[0])

    norm_avalanche_pdf_size = normalize_avalanche_pdf_size(mask_avalanche_s_0_bc,\
                                                           mask_avalanche_d_0_bc,\
                                                           mask_avalanche_t_0_bc,\
                                                           mask_avalanche_s_1_bc,\
                                                           mask_avalanche_d_1_bc,\
                                                           mask_avalanche_t_1_bc)

    print("linscore_list", linscore_list)
    print("coef_list", coef_list)
    print("ksdist_list", ksdist_list)

    norm_linscore_res = norm_linscore(linscore_list)
    norm_ksdist_res = norm_ksdist(ksdist_list)
    norm_coef_res = norm_coef(coef_list)
    norm_unique_states = ((np.unique(ca_result, axis=0).shape[0]) / ca_result.shape[1])

    print("norm_avalanche_pdf_size", norm_avalanche_pdf_size)
    print("norm_linscore_res", norm_linscore_res)
    print("norm_ksdist_res", norm_ksdist_res)
    print("norm_coef_res", norm_coef_res)
    print("norm_unique_states", norm_unique_states)

    fitness = norm_ksdist_res**2 + norm_unique_states + norm_avalanche_pdf_size + norm_linscore_res**2

    if fitness > 3.0:
      R_list = []
      R_list.append(calculate_comparison_ratio(avalanche_s_0))
      R_list.append(calculate_comparison_ratio(avalanche_d_0))
      R_list.append(calculate_comparison_ratio(avalanche_t_0))
      R_list.append(calculate_comparison_ratio(avalanche_s_1))
      R_list.append(calculate_comparison_ratio(avalanche_d_1))
      R_list.append(calculate_comparison_ratio(avalanche_t_1))
      print("R_list", R_list)
      norm_R_res = norm_comparison_ratio(R_list)
      print("norm_R_res", norm_R_res)
      fitness = fitness + norm_R_res

  val_dict = {}
  val_dict["norm_ksdist_res"] = norm_ksdist_res
  val_dict["norm_coef_res"] = norm_coef_res
  val_dict["norm_unique_states"] = norm_unique_states
  val_dict["norm_avalanche_pdf_size"] = norm_avalanche_pdf_size
  val_dict["norm_linscore_res"] = norm_linscore_res
  val_dict["norm_R_res"] = norm_R_res
  val_dict["fitness"] = fitness

  print("Fitness", fitness)
  return fitness, val_dict

def powerlaw_stats(data, args, fname, timestr):
  data.remove(max(data))
  d = np.array(data)

  fit = powerlaw.Fit(d, discrete=True)
  print()
  print("alpha", fit.power_law.alpha)
  print("xmin", fit.power_law.xmin)
  print("sigma", fit.power_law.sigma)
  print("KSdist", fit.power_law.D)
  print("fit.distribution_compare('power_law', 'exponential')", fit.distribution_compare('power_law', 'exponential', normalized_ratio=True))
  print("fit.distribution_compare('power_law', 'lognormal')", fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True))
  gof = goodness_of_fit(fit, data, xmin=fit.power_law.xmin)
  print("goodness_of_fit(fit, data)", gof)
  print()
  fig, ax = plt.subplots()

  fit.plot_pdf(color = "b", linewidth=2, ax =ax, label="Avalanche (samples=%d)"% len(d))
  fit.power_law.plot_pdf(color = "k", linestyle = "--", ax =ax, label=r"Fit ($\hat{\alpha}$="+"%.1f, $p$-value=%.2f)" % (fit.power_law.alpha, gof))

  ax.legend()
  ax.set_xlabel("$x$")
  ax.set_ylabel("$P(x)$")
  print("ax.get_xticklabels()", fname, ax.get_xticklabels(which="major"))
  if len(ax.get_xticklabels(which="major")) < 6:
    ax.set_xlim(right=100)
  plt.tight_layout()
  if fname:
    path_powerlaw_png = os.path.join(args.log_dir,"power_"+fname+timestr+".png")
    plt.savefig(path_powerlaw_png, format="png")
  plt.close('all')

  fig, ax = plt.subplots()
  pdf = np.bincount(data)
  pdf = pdf / sum(pdf)
  x = np.linspace(1,len(pdf),len(pdf))
  print(x[pdf > 0])
  print(pdf[pdf > 0])
  ax.scatter(x[pdf > 0], pdf[pdf > 0], c='b', label="Avalanche (samples=%d)"% len(data))
  fit.power_law.plot_pdf(color = "k", linestyle = "--", ax =ax, label=r"Fit ($\hat{\alpha}$="+"%.1f, $p$-value=%.2f)" % (fit.power_law.alpha, gof))

  ax.legend()
  ax.set_xlabel("$x$")
  ax.set_ylabel("$P(x)$")
  plt.tight_layout(w_pad=0.1)
  if fname:
    path_real_powerlaw_png = os.path.join(args.log_dir,"power_real_"+fname+timestr+".png")
    plt.savefig(path_real_powerlaw_png, format="png")
  plt.close('all')

def get_numbered_avalanches(x, value):
  x_value = (x == value).astype(np.int8)
  numbered_avalanches_x = measure.label(x_value, background=0)
  for i in range(numbered_avalanches_x.shape[0]):
    if numbered_avalanches_x[i,0] != 0 and numbered_avalanches_x[i,-1] != 0 and\
      numbered_avalanches_x[i,0] != numbered_avalanches_x[i,-1]:
      numbered_avalanches_x[numbered_avalanches_x == numbered_avalanches_x[i,-1]] = numbered_avalanches_x[i,0]

  return numbered_avalanches_x

def get_ava_rgb(numbered_avalanches):
  numbered_avalanches_rgb = np.full((numbered_avalanches.shape[0],numbered_avalanches.shape[1], 3), 255)
  numbered_avalanches_rgb[np.logical_and(numbered_avalanches % 6 == 0, numbered_avalanches != 0)] = np.array([255,0,0]).reshape((1,1,3))
  numbered_avalanches_rgb[numbered_avalanches % 6 == 1] = np.array([0,255,0]).reshape((1,1,3))
  numbered_avalanches_rgb[numbered_avalanches % 6 == 2] = np.array([0,0,255]).reshape((1,1,3))
  numbered_avalanches_rgb[numbered_avalanches % 6 == 3] = np.array([255,255,0]).reshape((1,1,3))
  numbered_avalanches_rgb[numbered_avalanches % 6 == 4] = np.array([255,0,255]).reshape((1,1,3))
  numbered_avalanches_rgb[numbered_avalanches % 6 == 5] = np.array([0,255,255]).reshape((1,1,3))

  return numbered_avalanches_rgb.astype(np.uint8)


def getarray_avalanche_size_duration_total(x, value, save_plot=True):
  numbered_avalanches = get_numbered_avalanches(x,value)
  # print(numbered_avalanches[:16,:16])

  if save_plot:
    if not os.path.exists("results"):
      os.makedirs("results")
    img = Image.fromarray(get_ava_rgb(numbered_avalanches)).resize((5*width,5*timesteps), Image.NEAREST)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    img.save("results/soc_ca_v2_numbered_"+timestr+"_"+str(value)+".png")

  number_of_avalanches = int(np.max(numbered_avalanches))
  avalanche_size = []
  avalanche_duration = []
  avalanche_total = []

  for avalanche_number in range(1,number_of_avalanches+1):
    avalanche = np.argwhere(numbered_avalanches == avalanche_number)
    if len(avalanche) > 0:
      avalanche_duration.append(len(np.unique(avalanche[:,0])))
      avalanche_size.append(len(np.unique(avalanche[:,1])))
      avalanche_total.append(len(avalanche))

  return avalanche_size, avalanche_duration, avalanche_total

def save_avalanche_plot(data, datatype_str, args, gen):
  fit = powerlaw.Fit(data, xmin=1, discrete= True)
  fig, ax = plt.subplots()

  try:
    fit.plot_pdf(color = "b", linewidth=2, ax=ax, label="Avalanche (samples=%d)"% len(data))
    fit.power_law.plot_pdf(color = "k", linestyle = "--", ax =ax, label=r"Fit ($\hat{\alpha}$="+"%.2f)" % (fit.power_law.alpha))
    print("fit.power_law.D", fit.power_law.D)
    print("fit.power_law.xmin", fit.power_law.xmin)

    pdf = np.bincount(data)[1:] if len(data) > 5 else np.array([1])
    pdf = pdf/sum(pdf)
  except:
    error_msg = "Error in producing this plot: ca_%06d_"%gen+datatype_str+".png"
    ax.set_title(error_msg)
    print(error_msg)

  ax.legend()
  ax.set_xlabel("$x$")
  ax.set_ylabel("$P(x)$")

  plt.tight_layout(w_pad=0.1)
  plt.savefig(os.path.join(args.log_dir,"ca_%06d_"%gen+datatype_str+".png"), format="png")
  plt.close("all")

def plot_ca_result(ca_result, args, gen):
  ca_result_img = ca_result.astype(np.uint8)*255

  img = Image.fromarray(ca_result_img).resize((5*width,5*timesteps), Image.NEAREST)
  img.save(os.path.join(args.log_dir,'ca_%06d.png' % gen))

  avalanche_s_0, avalanche_d_0, avalanche_t_0 = getarray_avalanche_size_duration_total(ca_result, 0, save_plot=False)
  avalanche_s_1, avalanche_d_1, avalanche_t_1 = getarray_avalanche_size_duration_total(ca_result, 1, save_plot=False)

  save_avalanche_plot(avalanche_s_0, "s_0", args, gen)
  save_avalanche_plot(avalanche_d_0, "d_0", args, gen)
  save_avalanche_plot(avalanche_t_0, "t_0", args, gen)
  save_avalanche_plot(avalanche_s_1, "s_1", args, gen)
  save_avalanche_plot(avalanche_d_1, "d_1", args, gen)
  save_avalanche_plot(avalanche_t_1, "t_1", args, gen)

def plot_ca_result_test(ca_result, args, gen):
  ca_result_img = 255 - ca_result.astype(np.uint8)*255

  img = Image.fromarray(ca_result_img).resize((5*width,5*timesteps), Image.NEAREST)
  img.save(os.path.join(args.log_dir,'test_ca_%06d.png' % gen))

  avalanche_s_0, avalanche_d_0, avalanche_t_0 = getarray_avalanche_size_duration_total(ca_result, 0, save_plot=False)
  avalanche_s_1, avalanche_d_1, avalanche_t_1 = getarray_avalanche_size_duration_total(ca_result, 1, save_plot=False)

  save_avalanche_plot(avalanche_s_0, "test_s_0", args, gen)
  save_avalanche_plot(avalanche_d_0, "test_d_0", args, gen)
  save_avalanche_plot(avalanche_t_0, "test_t_0", args, gen)
  save_avalanche_plot(avalanche_s_1, "test_s_1", args, gen)
  save_avalanche_plot(avalanche_d_1, "test_d_1", args, gen)
  save_avalanche_plot(avalanche_t_1, "test_t_1", args, gen)

  powerlaw_stats(avalanche_s_0, args, "avalanche_s_0", "")
  powerlaw_stats(avalanche_d_0, args, "avalanche_d_0", "")
  powerlaw_stats(avalanche_t_0, args, "avalanche_t_0", "")
  powerlaw_stats(avalanche_s_1, args, "avalanche_s_1", "")
  powerlaw_stats(avalanche_d_1, args, "avalanche_d_1", "")
  powerlaw_stats(avalanche_t_1, args, "avalanche_t_1", "")

def evaluate_nca(flat_weights, args, gen=None, test=None):
  nca = CriticalNCA()
  weight_shape_list, weight_amount_list, _ = utils.get_weights_info(nca.weights)
  shaped_weight = utils.get_model_weights(flat_weights, weight_amount_list,
                                          weight_shape_list)
  nca.dmodel.set_weights(shaped_weight)

  x = np.zeros((1, width,nca.channel_n),
               dtype=np.float32)
  np.random.seed(1)
  x[:,:,:1] = np.random.randint(2, size=(1,width,1))

  x_history = [x[0,:,0]]
  for t in range(timesteps-1):
    x = nca(x)
    x_history.append(x[0,:,0])

  x_history_arr = np.array(x_history)

  fitness, val_dict = evaluate_result(x_history_arr)

  if gen is not None:
    print("np.unique(x_history_arr)", np.unique(x_history_arr))
    plot_ca_result(x_history_arr, args, gen)

    with open(os.path.join(args.log_dir,'%06d.txt' % gen), "w") as f:
      f.write(str(val_dict))

  if test is not None:
    print("np.unique(x_history_arr)", np.unique(x_history_arr))
    plot_ca_result_test(x_history_arr, args, test)


  return -1*fitness, val_dict
