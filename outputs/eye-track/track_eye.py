import sys,json
sys.path.insert(0,'E:/cooking/fine-tuning/outputs/eye-track/python-libs')
import cv2
import numpy as np
from pathlib import Path
out=Path('E:/cooking/fine-tuning/outputs/eye-track')
cap=cv2.VideoCapture('E:/les_numeriques/3/YT TV LG/C3818.MP4')
frames=[]
for i in range(64):
 ok,f=cap.read()
 if not ok: raise RuntimeError('Missing frame')
 frames.append(f)
gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]
mask=np.zeros_like(gray[0]);mask[430:590,1735:1910]=255;mask[475:530,1760:1890]=0
points=cv2.goodFeaturesToTrack(gray[0],120,.015,6,mask=mask).astype(np.float32)
initial=points.copy();prev=points.copy();center=np.array([1824.,511.],np.float32)
positions=[center.tolist()];errors=[]; eye=np.array([[[1777.,511.]],[[1874.,514.]]],np.float32); eye0=eye.copy()
for i in range(1,64):
 nxt,status,err=cv2.calcOpticalFlowPyrLK(gray[i-1],gray[i],prev,None,winSize=(35,35),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
 back,s2,e2=cv2.calcOpticalFlowPyrLK(gray[i],gray[i-1],nxt,None,winSize=(35,35),maxLevel=3)
 good=(status.ravel()==1)&(s2.ravel()==1)&(np.linalg.norm(back-prev,axis=2).ravel()<1.5)
 initial=initial[good];nxt=nxt[good]
 mat,inliers=cv2.estimateAffinePartial2D(initial.reshape(-1,2),nxt.reshape(-1,2),method=cv2.RANSAC,ransacReprojThreshold=2)
 if mat is None: raise RuntimeError('Tracking failed '+str(i))
 eye,es,ee=cv2.calcOpticalFlowPyrLK(gray[i-1],gray[i],eye,None,winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,40,.001))
 p=center+(eye-eye0).reshape(-1,2).mean(axis=0);positions.append([round(float(p[0]),3),round(float(p[1]),3)])
 errors.append({'frame':i,'points':len(nxt),'inliers':int(inliers.sum())});prev=nxt
(out/'track.json').write_text(json.dumps({'times':[i/25 for i in range(64)],'values':positions,'quality':errors}))
tiles=[]
for i in [0,8,16,24,32,40,48,56,63]:
 f=frames[i].copy();p=tuple(round(v) for v in positions[i]);cv2.drawMarker(f,p,(0,255,255),cv2.MARKER_CROSS,24,2)
 crop=f[330:730,1650:2150].copy();cv2.putText(crop,str(i)+' / '+str(i/25)+'s',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2);tiles.append(crop)
cv2.imwrite(str(out/'tracking-check.jpg'),np.vstack([np.hstack(tiles[j:j+3]) for j in [0,3,6]]))
print(json.dumps({'first':positions[0],'last':positions[-1],'minPoints':min(q['points'] for q in errors),'minInliers':min(q['inliers'] for q in errors)}))

