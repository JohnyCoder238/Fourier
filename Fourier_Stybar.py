# import knihoven
from scipy import integrate
import numpy as np, matplotlib.pyplot as plt, scipy as sci, math

# kontrola kvadratické integrovatelnosti, vychází 16,38 - je tedy kvadraticky integrovatelná
square = (float(integrate.quad(lambda t: (np.exp(-2*t)-1)**2, 0,1)[0]) + float(integrate.quad(lambda t: 2**2,1,5)[0]))

T = 5 # perioda
omega = 2 * sci.pi / T  #úhlová frekvence
N = 3   # počet členů přes které je proveden součet
t = np.arange(0,14,0.02)

# výpočet a_0
a_0 = 2/T * (float(integrate.quad(lambda t: (np.exp(-2*t)-1), 0,1)[0]) + float(integrate.quad(lambda t: 2,1,5)[0]))
print("a_0: "+str(a_0))

a = []
b = []

# výpočet koeficientů a_n a b_n  pro N členů pro Fourierovu řadu
for n in range(1,N):
    a_n  = 2/T * (float(integrate.quad(lambda t: (np.exp(-2*t)-1)*np.cos(n*omega*t), 0,1)[0]) + float(integrate.quad(lambda t: 2 *np.cos(n*omega*t),1,5)[0]))   # výpočet a_n
    a.append(a_n)
    b_n = 2/T * (float(integrate.quad(lambda t: (np.exp(-2*t)-1)*np.sin(n*omega*t), 0,1)[0]) + float(integrate.quad(lambda t: 2 *np.sin(n*omega*t),1,5)[0]))    # výpočet b_n
    b.append(b_n)

def recursive_fourier(t, n=N-1):    # rekurzivní součet Fourierovy řady
    if int(n)==1:
        return a[n-1]*np.cos(omega*t*n)+b[n-1]*np.sin(omega*t*n)
    else:
        return (recursive_fourier(t, n-1) + a[n-1]*np.cos(omega*t*n)+b[n-1]*np.sin(omega*t*n))

L=T
b_sine = []

# výpočet koeficientu b_n pro N členů pro sinovou Fourierovu řadu
for n in range(1,N):
    b_n = 2/L * (float(integrate.quad(lambda t: ((np.exp(-2*t)-1)*np.sin(n*np.pi/L*t)), 0,1)[0]) + float(integrate.quad(lambda t: (2*np.sin(n*np.pi/L*t)),1,5)[0]))
    b_sine.append(b_n)
    
def recursive_sine_fourier(t, n=N-1):   # rekurzivní součet Fourierovy sinové řady
    if int(n)==1:
        return b_sine[n-1]*np.sin(t*np.pi/L*n)
    else:
        return (recursive_sine_fourier(t, n-1) + b_sine[n-1]*np.sin(t*np.pi/L*n))

# funkce pro výpočet po částech definované funkce
def make_piecewise(t):
    y = np.array([2 if x%5>1 else (np.exp(-2*(x%5))-1) for x in t])
    y[:-1][np.absolute(np.diff(y)) >= 0.8] = np.nan
    return y

# výpočet průměrů ve skocích
modulo_t = [x%5 if np.isclose(x%5,0) or np.isclose(x%5,1) else np.nan for x in t]
averages_t = t[~np.isnan(modulo_t)]
average_y_points = [np.mean([2,np.exp(-2*(x%5))-1]) for x in averages_t]
lower_y_points = [np.exp(-2*(x%5))-1 for x in averages_t]
upper_y_points = [2 for x in averages_t]

# graf Součet nekonečné Fourierovy řady
plt.plot(t,make_piecewise(t))
plt.plot(averages_t, average_y_points,'bo')
plt.plot(averages_t, lower_y_points,'bo', fillstyle='none')
plt.plot(averages_t, upper_y_points,'bo', fillstyle='none')
plt.title("Součet nekonečné Fourierovy řady")
plt.grid()
plt.xlabel("t")
plt.ylabel("y")
plt.show()

# funkce pro výpočet po částech definované funkce - lichá verze
def make_piecewise_odd(t):
    values=[]
    for x in t:
        if x%10<1:
            values.append(np.exp(-2*(x%10))-1)
        elif x%10>1 and x%10<5:
            values.append(2)
        elif x%10>5 and x%10<9:
            values.append(-2)
        elif x%10>9:
            values.append(-(np.exp(-2*((-x)%5))-1))
        else:
            values.append(np.nan)
    return values

# pomocná funkce pro výpočet průměrů ve skocích - lichá verze
def make_averages(t):
    values = []
    for x in t:
        if np.isclose(x%10,1):
            values.append(np.mean([2,np.exp(-2*(x%5))-1]))
        elif np.isclose(x%10,5):
            values.append(0)
        else:
            values.append(np.mean([-(np.exp(-2*((-x)%5))-1),-2]))
    return values

# samotný výpočet průměrů ve skocích
modulo_t = [x%5 if np.isclose(x%10,1) or np.isclose(x%10,5) or np.isclose(x%10,9) else np.nan for x in t]
averages_t = t[~np.isnan(modulo_t)]
average_y_points = make_averages(averages_t)
lower_y_points = [np.exp(-2*(x%5))-1 if np.isclose(x%10,1) else -2 for x in averages_t]
upper_y_points = [2 if not(np.isclose(x%10,9)) else -(np.exp(-2*((-x)%5))-1) for x in averages_t]

# graf Součet nekonečné Fourierovy sinové řady
plt.plot(t,make_piecewise_odd(t))
plt.plot(averages_t, average_y_points,'bo')
plt.plot(averages_t, lower_y_points,'bo', fillstyle='none')
plt.plot(averages_t, upper_y_points,'bo', fillstyle='none')
plt.title("Součet nekonečné Fourierovy sinové řady")
plt.grid()
plt.xlabel("t")
plt.ylabel("y")
plt.show()

# graf pro Součet prvních 3 členů Fourierovy řady
plt.title("Součet prvních 3 členů Fourierovy řady")
plt.grid()
plt.xlabel("t")
fourier = np.add(list(map(recursive_fourier, t)),a_0/2) # hodnoty Fourierovy funkce
plt.plot(t,fourier)
plt.show()

# graf pro Součet prvních 3 členů Fourierovy sinové řady
plt.title("Součet prvních 3 členů Fourierovy sinové řady")
plt.grid()
plt.xlabel("t")
fourier_sine = list(map(recursive_sine_fourier, t))  # hodnoty Fourierovy sinové řady
plt.plot(t,fourier_sine)
plt.show()

# výpočet prvních pěti členů amplitudového a fázového spektra Fourierovy řady
amplitudes = []
phases = []
amplitudes.append(np.abs(a_0/2))
phases.append(np.nan)

for n in range(1,7):
    a_n  = 2/T * (float(integrate.quad(lambda t: (np.exp(-2*t)-1)*np.cos(n*omega*t), 0,1)[0]) + float(integrate.quad(lambda t: 2 *np.cos(n*omega*t),1,5)[0]))   # výpočet a_n
    b_n = 2/T * (float(integrate.quad(lambda t: (np.exp(-2*t)-1)*np.sin(n*omega*t), 0,1)[0]) + float(integrate.quad(lambda t: 2 *np.sin(n*omega*t),1,5)[0]))    # výpočet b_n
    amplitude = np.sqrt(b_n**2+ a_n**2)
    amplitudes.append(amplitude)
    phase = np.arccos(a_n/amplitude) if b_n<0 else -np.arccos(a_n/amplitude)
    phases.append(phase)

# graf pro Amplitudové spektrum prvních 5 členů Fourierovy řady
plt.title("Amplitudové spektrum prvních 5 členů Fourierovy řady")
plt.scatter(np.arange(5),amplitudes[:5])
plt.xlabel("t")
plt.grid()
plt.show()

# graf pro Fázové spektrum prvních 5 členů Fourierovy řady
plt.title("Fázové spektrum prvních 5 členů Fourierovy řady")
plt.grid()
plt.xlabel("t")
plt.scatter(np.arange(7),phases[:7])
plt.xlim(-0.2,5.2)
plt.show()

# výpočet prvních pěti členů amplitudového a fázového spektra Fourierovy sinové řady
sine_amplitudes = []
sine_phases = []
sine_amplitudes.append(np.abs(a_0/2))
sine_phases.append(np.nan)

for n in range(1,7):
    b_n = 2/L * (float(integrate.quad(lambda t: ((np.exp(-2*t)-1)*np.sin(n*np.pi/L*t)), 0,1)[0]) + float(integrate.quad(lambda t: (2*np.sin(n*np.pi/L*t)),1,5)[0]))
    amplitude = np.sqrt(b_n**2)
    sine_amplitudes.append(amplitude)
    phase = np.arccos(0/amplitude) if b_n<0 else -np.arccos(0/amplitude)
    sine_phases.append(phase)

# graf Amplitudové spektrum prvních 5 členů sinové Fourierovy řady
plt.title("Amplitudové spektrum prvních 5 členů sinové Fourierovy řady")
plt.scatter(np.arange(5),sine_amplitudes[:5])
plt.xlabel("t")
plt.grid()
plt.show()

# graf Fázové spektrum prvních 5 členů sinové Fourierovy řady
plt.title("Fázové spektrum prvních 5 členů sinové Fourierovy řady")
plt.grid()
plt.xlabel("t")
plt.scatter(np.arange(7),sine_phases[:7])
plt.xlim(-0.2,5.2)
plt.show()