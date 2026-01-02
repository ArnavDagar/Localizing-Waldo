import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

''' Generate CDF and Avg Time Plots for Localization Results'''
fileStr = "C:/Users/Arnav/OneDrive/Documents/LegoPython/localization_perfF.csv"

plt.rcParams["figure.figsize"] = (1.5,1.5)

def generate_cdf(df, nParticles, sensor, metric, bins):
    '''generate_cdf(df, nParticles, sensor, metric) -> cdf
    Computes positon error cdf for the metric after filtering
    the dataframe for the given parameters'''
    idx = (df['nParticles'] == nParticles) & (df['sensor'] == sensor)
    count, bins_count = np.histogram(df[idx][metric], bins = bins)
    pdf = count/sum(count)
    cdf = np.cumsum(pdf)
    return cdf

def generate_cdf_plots(lCDFs,lLabels,titleStr,figStr,xLabel,bins):
    '''generate_pe_cdf_plots(lCDFs,lLabels)->None
    Plots the CDFs specified and adds given labels'''
    fig, ax = plt.subplots()
    ax.set_xlabel(xLabel)
    ax.set_ylabel('Probability <=')
    for i in range(len(lCDFs)):
        ax.plot(bins[1:], lCDFs[i], label= lLabels[i])
    ax.set_title(titleStr, fontsize = 10)
    #ax.set_xlim(left = 0, right = 80)
    #ax.set_ylim(bottom = 0, top = 0.5)
    plt.legend()
    plt.grid()
    plt.show(block=False)
    plt.savefig(figStr)


def generate_avg_time(df,sensor):
    '''generate_avg_time(df,nParticles,sensor)->avgTime
    computes avg time for different particle numbers for the given sensor'''
    idx = (df['sensor'] == sensor)
    grouped_df = df[idx].groupby('nParticles').mean()
    grouped_df = grouped_df.reset_index()
    return grouped_df['time(s)']
    
    
df = pd.read_csv(fileStr)
bins = np.arange(0,1210,10)


#Code to generate Weighted CDFs
p1MbPEcsCDF = generate_cdf(df, 1000000, 'cs', 'wPE(mm)',bins)
p1MbPEusCDF = generate_cdf(df, 1000000, 'us', 'wPE(mm)',bins)
p1MbPEfsCDF = generate_cdf(df, 1000000, 'fs', 'wPE(mm)',bins)
fPECDF = generate_cdf(df, 1000000, 'fs', 'fPE(mm)',bins)
rPECDF = generate_cdf(df, 1000000, 'fs', 'rPE(mm)',bins)
lCDFs = [p1MbPEcsCDF, p1MbPEusCDF, p1MbPEfsCDF, fPECDF, rPECDF]
lLabels = ["Color Sensor", "Ultrasonic Sensor", "Both Sensors", \
           "MidPoint (Baseline2)",  "Random (Baseline1)"]
titleStr = "Weighted Position Error CDFs, 1000,000 particles"
figStr = 'zp1MwPECDFs.jpg'
generate_cdf_plots(lCDFs,lLabels,titleStr,figStr,'mm',bins)

p1MbPEcsCDF = generate_cdf(df, 1000000, 'cs', 'bPE(mm)',bins)
p1MbPEusCDF = generate_cdf(df, 1000000, 'us', 'bPE(mm)',bins)
p1MbPEfsCDF = generate_cdf(df, 1000000, 'fs', 'bPE(mm)',bins)
fPECDF = generate_cdf(df, 1000000, 'fs', 'fPE(mm)',bins)
rPECDF = generate_cdf(df, 1000000, 'fs', 'rPE(mm)',bins)
lCDFs = [p1MbPEcsCDF, p1MbPEusCDF, p1MbPEfsCDF, fPECDF, rPECDF]
lLabels = ["Color Sensor", "Ultrasonic Sensor", "Both Sensors", \
           "MidPoint (Baseline2)",  "Random (Baseline1)"]
titleStr = "Best Particle Position Error CDFs, 1000,000 particles"
figStr = 'zp1MbPECDFs.jpg'
generate_cdf_plots(lCDFs,lLabels,titleStr,figStr,'mm',bins)


# Generate color sensor cdfs for different particle numbers
p1HbPEcsCDF = generate_cdf(df, 100, 'cs', 'bPE(mm)',bins)
p1KbPEcsCDF = generate_cdf(df, 1000, 'cs', 'bPE(mm)',bins)
p10KbPEcsCDF = generate_cdf(df, 10000, 'cs', 'bPE(mm)',bins)
p100KbPEcsCDF = generate_cdf(df, 100000, 'cs', 'bPE(mm)',bins)
p1MbPEcsCDF = generate_cdf(df, 1000000, 'cs', 'bPE(mm)',bins)

lCDFs = [p1HbPEcsCDF, p1KbPEcsCDF, p10KbPEcsCDF, p100KbPEcsCDF, p1MbPEcsCDF]
lLabels = ["100 Particles", "1K Particles", "10K Particles", \
           "100K Particles",  "1M Particles"]
titleStr = "Best Particle Position Error CDFs, Color Sensor"
figStr = 'CSbPECDFs.jpg'
generate_cdf_plots(lCDFs,lLabels,titleStr,figStr,'mm',bins)

# Generate Ultrasonic sensor cdfs for different particle numbers
p1HbPEusCDF = generate_cdf(df, 100, 'us', 'bPE(mm)',bins)
p1KbPEusCDF = generate_cdf(df, 1000, 'us', 'bPE(mm)',bins)
p10KbPEusCDF = generate_cdf(df, 10000, 'us', 'bPE(mm)',bins)
p100KbPEusCDF = generate_cdf(df, 100000, 'us', 'bPE(mm)',bins)
p1MbPEusCDF = generate_cdf(df, 1000000, 'us', 'bPE(mm)',bins)

lCDFs = [p1HbPEusCDF, p1KbPEusCDF, p10KbPEusCDF, p100KbPEusCDF, p1MbPEusCDF]
lLabels = ["100 Particles", "1K Particles", "10K Particles", \
           "100K Particles",  "1M Particles"]
titleStr = "Best Particle Position Error CDFs, Ultrasonic Sensor"
figStr = 'USbPECDFs.jpg'
generate_cdf_plots(lCDFs,lLabels,titleStr,figStr,'mm',bins)

# Generate Both sensor cdfs for different particle numbers
p1HbPEfsCDF = generate_cdf(df, 100, 'fs', 'bPE(mm)',bins)
p1KbPEfsCDF = generate_cdf(df, 1000, 'fs', 'bPE(mm)',bins)
p10KbPEfsCDF = generate_cdf(df, 10000, 'fs', 'bPE(mm)',bins)
p100KbPEfsCDF = generate_cdf(df, 100000, 'fs', 'bPE(mm)',bins)
p1MbPEfsCDF = generate_cdf(df, 1000000, 'fs', 'bPE(mm)',bins)

lCDFs = [p1HbPEfsCDF, p1KbPEfsCDF, p10KbPEfsCDF, p100KbPEfsCDF, p1MbPEfsCDF]
lLabels = ["100 Particles", "1K Particles", "10K Particles", \
           "100K Particles",  "1M Particles"]
titleStr = "Best Particle Position Error CDFs, Both Sensors"
figStr = 'FSbPECDFs.jpg'
generate_cdf_plots(lCDFs,lLabels,titleStr,figStr,'mm',bins)


bins = np.arange(360)
# Generate color sensor cdfs for different particle numbers
p1HbHEcsCDF = generate_cdf(df, 100, 'cs', 'bHE(degree)',bins)
p1KbHEcsCDF = generate_cdf(df, 1000, 'cs', 'bHE(degree)',bins)
p10KbHEcsCDF = generate_cdf(df, 10000, 'cs', 'bHE(degree)',bins)
p100KbHEcsCDF = generate_cdf(df, 100000, 'cs', 'bHE(degree)',bins)
p1MbHEcsCDF = generate_cdf(df, 1000000, 'cs', 'bHE(degree)',bins)

lCDFs = [p1HbHEcsCDF, p1KbHEcsCDF, p10KbHEcsCDF, p100KbHEcsCDF, p1MbHEcsCDF]
lLabels = ["100 Particles", "1K Particles", "10K Particles", \
           "100K Particles",  "1M Particles"]
titleStr = "Best Particle Heading Error CDFs, Color Sensor"
figStr = 'CSbHECDFs.jpg'
generate_cdf_plots(lCDFs,lLabels,titleStr,figStr,'degree',bins)

# Generate Ultrasonic sensor cdfs for different particle numbers
p1HbHEusCDF = generate_cdf(df, 100, 'us', 'bHE(degree)',bins)
p1KbHEusCDF = generate_cdf(df, 1000, 'us', 'bHE(degree)',bins)
p10KbHEusCDF = generate_cdf(df, 10000, 'us', 'bHE(degree)',bins)
p100KbHEusCDF = generate_cdf(df, 100000, 'us', 'bHE(degree)',bins)
p1MbHEusCDF = generate_cdf(df, 1000000, 'us', 'bHE(degree)',bins)

lCDFs = [p1HbHEusCDF, p1KbHEusCDF, p10KbHEusCDF, p100KbHEusCDF, p1MbHEusCDF]
lLabels = ["100 Particles", "1K Particles", "10K Particles", \
           "100K Particles",  "1M Particles"]
titleStr = "Best Particle Heading Error CDFs, Ultrasonic Sensor"
figStr = 'USbHECDFs.jpg'
generate_cdf_plots(lCDFs,lLabels,titleStr,figStr,'degree',bins)

# Generate Both sensor cdfs for different particle numbers
p1HbHEfsCDF = generate_cdf(df, 100, 'fs', 'bHE(degree)',bins)
p1KbHEfsCDF = generate_cdf(df, 1000, 'fs', 'bHE(degree)',bins)
p10KbHEfsCDF = generate_cdf(df, 10000, 'fs', 'bHE(degree)',bins)
p100KbHEfsCDF = generate_cdf(df, 100000, 'fs', 'bHE(degree)',bins)
p1MbHEfsCDF = generate_cdf(df, 1000000, 'fs', 'bHE(degree)',bins)

lCDFs = [p1HbHEfsCDF, p1KbHEfsCDF, p10KbHEfsCDF, p100KbHEfsCDF, p1MbHEfsCDF]
lLabels = ["100 Particles", "1K Particles", "10K Particles", \
           "100K Particles",  "1M Particles"]
titleStr = "Best Particle Heading Error CDFs, Both Sensors"
figStr = 'FSbHECDFs.jpg'
generate_cdf_plots(lCDFs,lLabels,titleStr,figStr,'degree',bins)


fig, ax = plt.subplots()
ax.set_xlabel('nParticles')
ax.set_ylabel('time(s)')
csAvgTime = generate_avg_time(df,'cs')
usAvgTime = generate_avg_time(df,'us')
fsAvgTime = generate_avg_time(df,'fs')
#p = ['1H', '1K', '10K', '100K', '1M']
p = [100, 1000, 10000, 100000, 1000000]
#NVAvgTime = [0.026313, 0.150346, 1.761899, 49.808301]
#p = [100, 1000, 10000, 100000]
lData = [csAvgTime,usAvgTime,fsAvgTime]  
lLabels = ["Color Sensor", "Ultrasonic Sensor", "Both Sensors"]
for i in range(3):
    #ax.plot(p, lData[i], label= lLabels[i], marker = 'o')
    ax.plot(p, lData[i], marker = 'o')
    #ax.plot(p, lData[i][:4], label= "Vectorized", marker = 'o')
#ax.plot(p, NVAvgTime, label= "Non Vectorized", marker = 'o')
#ax.set_title('Avg time to localize', fontsize = 10)
ax.set_xlim(left = 0, right = 11000)
ax.set_ylim(bottom = 0, top = 0.2)
#plt.legend(loc='lower right')
#plt.legend()
plt.show(block=False)
plt.grid()
plt.savefig('v_vs_nv.jpg')


fig, ax = plt.subplots()
ax.set_xlabel('nParticles')
ax.set_ylabel('time(s)')

NVAvgTime = [0.026313, 0.150346, 1.761899, 49.808301]
p = [100, 1000, 10000, 100000]
ax.plot(p, NVAvgTime, marker = 'o')
ax.set_title('Avg time to localize (non vectorized)', fontsize = 10)
#ax.set_xlim(left = 0, right = 11000)
#ax.set_ylim(bottom = 0, top = 0.2)
#plt.legend(loc='lower right')
plt.show(block=False)
plt.grid()
plt.savefig('nv.jpg')




