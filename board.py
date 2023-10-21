import socket

import numpy as np
import pygame
import sys
#调用常用关键字常量
from pygame.locals import QUIT,KEYDOWN

#初始化pygame
pygame.init()
#获取对显示系统的访问，并创建一个窗口screen
screen = pygame.display.set_mode((684,684))
screen_color=[238,154,73]#设置画布颜色,[238,154,73]对应为棕黄色
line_color = [0,0,0]#设置线条黑色

HOST = ''
PORT = 10888
s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(1)
conn, addr = s.accept()
print('Client\'s Address:',addr)



def find_pos():#找到显示的可以落子的位置
    x, y = pygame.mouse.get_pos()
    #print(x,y)
    for i in range(27,684,90):
        for j in range(27,684,90):
            L1=i-45
            L2=i+45
            R1=j-45
            R2=j+45
            if x>=L1 and x<=L2 and y>=R1 and y<=R2:
                return i,j
    return x,y


over_pos=[]#表示已经落子的位置
white_color=[255,255,255]#白棋颜色
black_color=[0,0,0]#黑棋颜色



def check_over_pos(x,y,over_pos):#检查当前的位置是否已经落子
    for val in over_pos:
        if val[0][0]==x and val[0][1]==y:
            return False
    return True#表示没有落子
flag=False
tim=0


def check_win(over_pos):#判断五子连心
    mp=np.zeros([8,8],dtype=int)
    for val in over_pos:
        x=int((val[0][0]-27)/90)
        y=int((val[0][1]-27)/90)
        if val[1]==white_color:
            mp[x][y]=2#表示白子
        else:
            mp[x][y]=1#表示黑子

    for i in range(8):
        pos1=[]
        pos2=[]
        for j in range(8):
            if mp[i][j]==1:
                pos1.append([i,j])
            else:
                pos1=[]
            if mp[i][j]==2:
                pos2.append([i,j])
            else:
                pos2=[]
            if len(pos1)>=5:#五子连心
                return [1,pos1]
            if len(pos2)>=5:
                return [2,pos2]

    for j in range(8):
        pos1=[]
        pos2=[]
        for i in range(8):
            if mp[i][j]==1:
                pos1.append([i,j])
            else:
                pos1=[]
            if mp[i][j]==2:
                pos2.append([i,j])
            else:
                pos2=[]
            if len(pos1)>=5:
                return [1,pos1]
            if len(pos2)>=5:
                return [2,pos2]
    for i in range(8):
        for j in range(8):
            pos1=[]
            pos2=[]
            for k in range(8):
                if i+k>=8 or j+k>=8:
                    break
                if mp[i+k][j+k]==1:
                    pos1.append([i+k,j+k])
                else:
                    pos1=[]
                if mp[i+k][j+k]==2:
                    pos2.append([i+k,j+k])
                else:
                    pos2=[]
                if len(pos1)>=5:
                    return [1,pos1]
                if len(pos2)>=5:
                    return [2,pos2]
    for i in range(8):
        for j in range(8):
            pos1=[]
            pos2=[]
            for k in range(8):
                if i+k>=8 or j-k<0:
                    break
                if mp[i+k][j-k]==1:
                    pos1.append([i+k,j-k])
                else:
                    pos1=[]
                if mp[i+k][j-k]==2:
                    pos2.append([i+k,j-k])
                else:
                    pos2=[]
                if len(pos1)>=5:
                    return [1,pos1]
                if len(pos2)>=5:
                    return [2,pos2]
    return [0,[]]


def init():
    screen.fill(screen_color)  # 清屏
    for i in range(27, 684, 90):
        # 先画竖线
        if i == 27 or i == 657:  # 边缘线稍微粗一些
            pygame.draw.line(screen, line_color, [i, 27], [i, 657], 4)
        else:
            pygame.draw.line(screen, line_color, [i, 27], [i, 657], 2)
        # 再画横线
        if i == 27 or i == 657:  # 边缘线稍微粗一些
            pygame.draw.line(screen, line_color, [27, i], [657, i], 4)
        else:
            pygame.draw.line(screen, line_color, [27, i], [657, i], 2)

    for val in over_pos:  # 显示所有落下的棋子
        pygame.draw.circle(screen, val[1], val[0], 20, 0)


while True:#不断训练刷新画布
    for event in pygame.event.get():#获取事件，如果鼠标点击右上角关闭按钮，关闭
        if event.type in (QUIT,KEYDOWN):
            sys.exit()

    init()
    res = check_win(over_pos)
    if res[0] != 0:
        for pos in res[1]:
            pygame.draw.rect(screen, [238, 48, 167], [pos[0] * 44 + 27 - 22, pos[1] * 90 + 27 - 22, 90, 90], 2, 1)
        pygame.display.update()  # 刷新显示
        break  # 游戏结束，停止下面的操作

    if len(over_pos) % 2 == 0:  # 黑子
        data = conn.recv(2048)
        data = data.decode('utf-8')
        print(data)
        data = int(data)
        y=int(data/8)
        x=int(data%8)
        y=7-y

        x=x*90+27
        y=y*90+27
        over_pos.append([[x, y], black_color])


    else:
        x=0
        y=0
        x,y=find_pos()
        n_x = (int)((x + 17) / 90)
        n_y = 7-(int)((y + 17) / 90)
        pygame.draw.rect(screen, [0, 229, 238], [x - 45, y - 45, 90, 90], 2, 1)

        keys_pressed = pygame.mouse.get_pressed()

        if keys_pressed[0] and tim == 0:
            flag = True
            if check_over_pos(x, y, over_pos):  # 判断是否可以落子，再落子
                    over_pos.append([[x, y], white_color])
                    tans = str(n_x + n_y * 8)
                    conn.sendall(tans.encode('utf-8'))

    init()
    if flag:
        tim += 1
    if tim % 200 == 0:  # 延时200ms
        flag = False
        tim = 0

    pygame.display.update()



