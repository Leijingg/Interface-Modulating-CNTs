import numpy as np
from scipy.optimize import least_squares
import xlrd

def realimag(array):
    return np.array([(x.real, -x.imag) for x in array])
def func(x,p):
    s,u,sigma,t=p #s:static dielectric constant;u:optical dielectricconstant;sigma:conductivity;t:relaxation time
    # s:静电介电常数;u:光学介电常数;sigma:电导率;t:弛豫时间
    d=complex(0,1)
    o=8.854187817*10**(-12)
    print('//////////////////////////////////////', s,u,t,x,p)
    print('sigma',sigma)
    return realimag(u+(s-u)/(1+x*x**t*t)-np.dot(d,np.dot(np.dot(x,t),np.divide((s-u),(1+np.dot(np.dot(np.dot(x,x),t),t)))))-np.dot(d,np.divide(sigma,np.dot(x,o))))
    #极化公式
def condut_loss_result(x,p):
    s,u,sigma,t=p
    o=8.854187817*10**(-12)
    print('//////////////////////////////////////', s,u,t,x,p)
    return np.divide(sigma,np.dot(x,o))
    #电导公式

def residuals(p,y,x):
    return (realimag(np.array(y)) - func(x, p)).flatten()

p0 = [20,10,1,10**-12]
data=xlrd.open_workbook('I:\Desktop\A.xlsx')  #read file 填写自己的文件路径

data_number=200
groups=20
table = data.sheet_by_name(u'Sheet1')
fcost=0
fsig=0 #fitting conductivity    #配件电导率
ft=0   #fitting relaxation time #拟合松弛时间
fplsq=[]
fs=0 #fitting static dielectric constant  #拟合静电介电常数
fu=0 #fitting optical dielectric constant #拟合光学介电常数
fconduct_loss=0 #fitting conducting loss  #配件进行损失
frelaxation_loss=0 #fitting polarization loss #合适的极化损失
for i in range(1,int(groups)+1):end=i*(int(data_number/groups))
start=end-int(data_number/groups)
print("Reading data group",i,": from ",start," to ",end)
xdata=table.col_values(0)[start:end]
ydata_1=table.col_values(2)[start:end] #imaginary permitivity #虚电容率
ydata_2=table.col_values(1)[start:end] #real permitivity #真正的电容率
ydata=[]
for m in range(int(data_number/groups)):xdata[m]=xdata[m]*10**9  # GHz
ydata.append(complex(ydata_2[m],-ydata_1[m]))
plsq = least_squares(residuals, p0,bounds=([0,0,0,0],[200,200,200,10**-1]),args=(ydata, xdata),max_nfev=100000)

fplsq.append(plsq)
print(plsq.x[0]," ",plsq.x[1]," ",plsq.x[2]," ",plsq.x[3]," ")
conduct_loss=np.mean(condut_loss_result(xdata,plsq.x))
print("Conduct_loss ",i,":",conduct_loss)
fconduct_loss += conduct_loss
relaxation_loss=np.mean(ydata_1)-conduct_loss
print('fconduct_loss:',np.mean(ydata_1))
print("Relaxation_loss",i,":",relaxation_loss)
print("虚数Imaginary:",np.mean(ydata_1),", Conduct Loss:",conduct_loss,",Relaxation Loss:",relaxation_loss)
print(np.mean(xdata)/10**9," ",np.mean(ydata_1)," ",conduct_loss,"",relaxation_loss)
frelaxation_loss += relaxation_loss
fsig=fsig+plsq.x[2]
fs=fs+plsq.x[0]
fu=fu+plsq.x[1]
ft=ft+plsq.x[3]
fcost=plsq.cost+fcost
#print("xdata and ydata",ydata,xdata)

