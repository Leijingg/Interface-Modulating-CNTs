import numpy as np
from scipy.optimize import least_squares
import xlrd

def realimag(array):
    return np.array([(x.real, -x.imag) for x in array])
def func(x,p):
    s,u,sigma,t=p #s:static dielectric constant;u:optical dielectricconstant;sigma:conductivity;t:relaxation time
    d=complex(0,1)
    o=8.854187817*10**(-12)
    return realimag(u+(s-u)/(1+x*x**t*t)-np.dot(d,np.dot(np.dot(x,t),np.divide((s-u),(1+np.dot(np.dot(np.dot(x,x),t),t)))))-np.dot(d,np.divide(sigma,np.dot(x,o))))

def condut_loss_result(x,p):
    s,u,sigma,t=p
    o=8.854187817*10**(-12)
    return np.divide(sigma,np.dot(x,o))

def residuals(p,y,x):
    return (realimag(np.array(y)) - func(x, p)).flatten()

p0 = [20,10,1,10**-12]
data=xlrd.open_workbook('I:\Desktop\A.xlsx')  #read file

data_number=200
groups=20
table = data.sheet_by_name(u'Sheet1')
fcost=0
fsig=0 #fitting conductivity
ft=0 #fitting relaxation time
fplsq=[]
fs=0 #fitting static dielectric constant
fu=0 #fitting optical dielectric constant
fconduct_loss=0 #fitting conducting loss
frelaxation_loss=0 #fitting polarization loss
for i in range(1,int(groups)+1):end=i*(int(data_number/groups))
start=end-int(data_number/groups)
#print("Reading data group",i,": from ",start," to ",end)
xdata=table.col_values(0)[start:end]
ydata_1=table.col_values(2)[start:end] #imaginary permitivity
ydata_2=table.col_values(1)[start:end] #real permitivity
ydata=[]
for m in range(int(data_number/groups)):xdata[m]=xdata[m]*10**9  # GHz
ydata.append(complex(ydata_2[m],-ydata_1[m]))
plsq = least_squares(residuals, p0,bounds=([0,0,0,0],[200,200,200,10**-1]),args=(ydata, xdata),max_nfev=100000)
fplsq.append(plsq)
print(plsq.x[0]," ",plsq.x[1]," ",plsq.x[2]," ",plsq.x[3]," ")
conduct_loss=np.mean(condut_loss_result(xdata,plsq.x))
#print("Conduct_loss ",i,":",conduct_loss)
fconduct_loss += conduct_loss
relaxation_loss=np.mean(ydata_1)-conduct_loss
#print("Relaxation_loss",i,":",relaxation_loss)
#print("Imaginary:",np.mean(ydata_1),", Conduct Loss:",conduct_loss,",Relaxation Loss:",relaxation_loss)
#print(np.mean(xdata)/10**9," ",np.mean(ydata_1)," ",conduct_loss,"",relaxation_loss)
frelaxation_loss += relaxation_loss
fsig=fsig+plsq.x[2]
fs=fs+plsq.x[0]
fu=fu+plsq.x[1]
ft=ft+plsq.x[3]
fcost=plsq.cost+fcost