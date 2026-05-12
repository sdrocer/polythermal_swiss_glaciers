pro CONNECT_SERVER, dir, dirout, $
  username=username, server=server, $
  dir_remote=dir_remote
  compile_opt idl2

  ; -------------------------------------------------------
  ; Server connection parameters - edit username if needed
  if not keyword_set(username)   then username   = 'jabeer'
  if not keyword_set(server)     then server     = 'vierzack06.ethz.ch'
  if not keyword_set(dir_remote) then dir_remote = '/scratch_net/vierzack06_fourth/mhuss/MassBalance_all/'
  ; -------------------------------------------------------

  mount_base = '/tmp/idl_firn_mounts/'
  dir    = mount_base + 'massbalance/'
  dirout = '/Users/janoschbeer/Library/Mobile Documents/com~apple~CloudDocs/PhD/projects/polythermal_swiss_glaciers/new_data/firn_results/'

  ; create local mount point for input and local output folder
  spawn, 'mkdir -p ' + dir
  spawn, 'mkdir -p "' + dirout + '"'

  ; unmount silently first (handles case where already mounted)
  spawn, 'umount ' + dir + ' 2>/dev/null'

  ; mount remote input directory via sshfs
  print, 'Connecting to ' + username + '@' + server + ' ...'
  spawn, 'sshfs ' + username + '@' + server + ':' + dir_remote + ' ' + dir

  ; wait until mount is ready before returning (avoids race condition)
  max_wait = 15  ; seconds
  t0 = systime(1)
  spawn, 'ls ' + dir, ls_result
  while ls_result[0] eq '' do begin
     if systime(1) - t0 gt max_wait then begin
        print, 'ERROR: sshfs mount timed out after ' + strtrim(max_wait,2) + ' s.'
        stop
     endif
     wait, 0.5
     spawn, 'ls ' + dir, ls_result
  endwhile

  print, 'Mounted:'
  print, '  Input:  ' + dir + '  ->  ' + dir_remote
  print, '  Output: ' + dirout + '  (local)'

end  ; {CONNECT_SERVER}


pro DISCONNECT_SERVER
  compile_opt idl2

  mount_base = '/tmp/idl_firn_mounts/'
  spawn, 'umount ' + mount_base + 'massbalance/'
  print, 'Server directories unmounted.'

end  ; {DISCONNECT_SERVER}


pro READ_AGR, fn, da, header=header, xx=xx, yy=yy, ncols=ncols, nrows=nrows, $
              xllcorner=xllcorner, yllcorner=yllcorner, $
              cellsize=cellsize, nodata_value=nodata_value
header=strarr(6) & openr,1, fn & readf,1, header
ncols=long(strmid(header(0),6,40)) & nrows=long(strmid(header(1),6,40))
da=dblarr(ncols,nrows) & readf,1, da & close, 1
xllcorner=double(strmid(header(2),10,40))
yllcorner=double(strmid(header(3),10,40))
cellsize=double(strmid(header(4),9,40))
nodata_value=double(strmid(header(5),13,40))
a=cellsize/2d & xx=lindgen(ncols)*cellsize+xllcorner+a
yy=lindgen(nrows)*cellsize+yllcorner+a
da=rotate(da,7)		; lower left corner is da(0,0)

END  ; {READ_AGR}

pro WRITE_AGR, fn, da, header=header, xx=xx, yy=yy, ncols=ncols, nrows=nrows, $
              xllcorner=xllcorner, yllcorner=yllcorner, $
              cellsize=cellsize, nodata_value=nodata_value, format=format

n=size(da) & dar=rotate(da,7)

if KEYWORD_SET(format) then $
  format=strmid(format,0,1)+strtrim(n(1),2)+strmid(format,1,40)

if KEYWORD_SET(header) then begin
    openw, 4, fn
    for i=0,5 do printf, 4, header(i)
    if KEYWORD_SET(format) then $
      for i=0l,n(2)-1 do printf, 4, dar(*,i), fo=format else $
      for i=0l,n(2)-1 do printf, 4, dar(*,i)
    close, 4
endif else begin
    if not KEYWORD_SET(xllcorner) then if KEYWORD_SET(xx) then $
      xllcorner=xx(0)-(xx(1)-xx(0))/2 else $
      stop, '%% WRITE_AGR :  XLLCORNER is not defined'
    if not KEYWORD_SET(yllcorner) then if KEYWORD_SET(yy) then $
      yllcorner=yy(0)-(yy(1)-yy(0))/2 else $
      stop, '%% WRITE_AGR :  YLLCORNER is not defined'
    if not KEYWORD_SET(cellsize) then $
      stop, '%% WRITE_AGR :  CELLSIZE is not defined'
    if not KEYWORD_SET(nodata_value) then $
      stop, '%% WRITE_AGR :  NODATA_VALUE is not defined'
    if not KEYWORD_SET(ncols) then ncols=n(1)
    if not KEYWORD_SET(nrows) then nrows=n(2)
    openw, 4, fn
    printf, 4, 'NCOLS       ', strtrim(ncols,2)
    printf, 4, 'NROWS       ', strtrim(nrows,2)
    printf, 4, 'XLLCORNER   ', xllcorner
    printf, 4, 'YLLCORNER   ', yllcorner
    printf, 4, 'CELLSIZE    ', cellsize
    printf, 4, 'NODATA_VALUE', nodata_value
    if KEYWORD_SET(format) then $
      for i=0l,n(2)-1 do printf, 4, dar(*,i), fo=format else $
      for i=0l,n(2)-1 do printf, 4, dar(*,i)
    close, 4
endelse

END  ; {WRITE_AGR}


; -------------------------------------------------------------------------
; -------------------------------------------------------------------------

; evaluate changes in firn coverage for selected glaciers

; connect to ETH server and set dir/dirout via sshfs mounts
; (server details and remote paths are defined in server_connect.pro)
CONNECT_SERVER, dir, dirout


glacier=['corvatsch','felskinn','hohsaas','tortin','sexrouge','alphubel']

tran=[1970,2026]  ; time range to be evaluated

time_firn=20    ; years for firn accumulation to become ice - not entirely sure how to set that

yrout=[1980,1990,2000,2010,2014,2019,2022,2024,2026]   ; arbitrary years for outputting firn thickness distribution 

; ----------------------------------

ng=n_elements(glacier)
len=tran(1)-tran(0)+1
noval=-9999

; -------------------------------------------------------------
; loop over glaciers
for g=0,ng-1 do begin

; read mass balance grids
READ_AGR,dir+glacier(g)+'/massbalance/results/massbalmeas_'+string(tran(1),fo='(i4)')+'.grid',b,ncols=nc,nrows=nr,header=head
glacier_mask = (b ne noval)  ; 1 = on-glacier, 0 = off-glacier (determined once from reference year)
bal=dblarr(len,nc,nr)+noval
for i=0,len-1 do begin
   a=findfile(dir+glacier(g)+'/massbalance/results/massbalmeas_'+string(tran(0)+i,fo='(i4)')+'.grid')
   if a(0) ne '' then begin
      READ_AGR,a(0),b & bal(i,*,*)=b
   endif
endfor

; evaluate / aggregate
firn=dblarr(nc,nr)   ; array for cumulating up firn (in m w.e.); 0 is ablation area
ageacc=dblarr(len,nc,nr)+noval              ; array for age stamp of firn layers
firnyear=ageacc                             ; array for annual firn layer remaining
firntot=ageacc                              ; array for storing total firn thickness at every time step

age=firn+noval              ; average age of firn layers - final evaluation
time_since_firn=firn+noval  ; years since firn disappeared (only where duration >= 2yr) - final evaluation
firn_duration=intarr(nc,nr)  ; consecutive years with firn per cell (for 2-year criterion)
year_firn_lost=firn+noval    ; year in which firn last disappeared at each cell
firn_area_ts=lonarr(len)     ; annual count of firn cells (for time-series output)

; loop over years
for i=0,len-1 do begin
   b=bal(i,*,*)
   ii=where(b gt 0,ci)   ; accumulation in that year
   jj=where(b le 0 and b ne noval and firn gt 0,cj)  ; ablation over firn area in that year 

   ; simple assessment - just cumulating accumulation
   firnlast=firn
   if ci gt 0 then firn(ii)=firn(ii)+b(ii)
   if cj gt 0 then firn(jj)=firn(jj)+b(jj)
   kk=where(glacier_mask eq 0,ck) & if ck gt 0 then firn(kk)=noval
   kk=where(firn lt 0 and firn gt -100,ck) & firn(kk)=0

   ; track firn duration and detect disappearance (minimum 2-year criterion)
   ll=where(firnlast gt 0 and firnlast ne noval and firn eq 0, cl)
   if cl gt 0 then begin
      valid=where(firn_duration(ll) ge 2, cvalid)
      if cvalid gt 0 then year_firn_lost(ll(valid))=tran(0)+i
      firn_duration(ll)=0
   endif
   mm=where(firn gt 0 and firn ne noval, cm)
   if cm gt 0 then firn_duration(mm)=firn_duration(mm)+1

   ; detailed assessment - tracking annual accumulation layers and storing age
   c=dblarr(nc,nr)+noval & d=c & e=c
   if ci gt 0 then c(ii)=b(ii) & firnyear(i,*,*)=c    ; filling up
   if ci gt 0 then e(ii)=i+tran(0) & ageacc(i,*,*)=e    ; tag year

      ; emtying
   if cj gt 0 then begin
      ; go through all cells, maybe inefficiently coded!
      for cc=0,nc-1 do begin
         for rr=0,nr-1 do begin
            for j=i-1,0,-1 do begin ; step through all years backwards
               if b(0,cc,rr) le 0 and b(0,cc,rr) ne noval and firnlast(cc,rr) gt 0 then begin
                  if abs(b(0,cc,rr)) gt firnyear(j,cc,rr) then begin
                     firnyear(j,cc,rr)=0 & ageacc(j,cc,rr)=noval
                  endif else begin
                     firnyear(j,cc,rr)=firnyear(j,cc,rr)+b(0,cc,rr) 
                     goto, endsearch
                  endelse
               endif   
            endfor
            endsearch:
         endfor
      endfor
   endif


; checking for entries in firnyear and ageacc that are too old (firn
; already transformed to ice)
ii=where(ageacc lt i+tran(0)-time_firn and ageacc ne noval,ci)
if ci gt 0 then firnyear(ii)=0
if ci gt 0 then ageacc(ii)=noval

; compute current firn thickness at every cell
for cc=0,nc-1 do begin
   for rr=0,nr-1 do begin
      if firn(cc,rr) ge 0 then begin ; only for cells on glacier
         a=firnyear(0:i,cc,rr) & ii=where(a ne noval,ci)
         if ci gt 0 then firntot(i,cc,rr)=total(a(ii))       
      endif
   endfor
endfor

; record firn cell count for this year (only cells with >= 2 consecutive years of firn)
ii=where(firn_duration ge 2,ci) & firn_area_ts(i)=ci

endfor    ; loop over years to be evaluated

; write annual firn area CSV
fn_csv = dirout + 'firn_area_annual_' + glacier(g) + '.csv'
openw, 9, fn_csv
printf, 9, 'year,firn_cells'
for i=0,len-1 do printf, 9, string(tran(0)+i,fo='(i4)') + ',' + strtrim(firn_area_ts(i),2)
close, 9

; -----------------------
; finalize evaluation, write out grids

; compute avg age of firn
ii=where(firn gt 0,ci)    
if ci gt 0 then begin
   for cc=0,nc-1 do begin
      for rr=0,nr-1 do begin
         if firn(cc,rr) gt 0 then begin
            a=ageacc(*,cc,rr) & kk=where(a ne noval,ck)
            if ck gt 0 then begin
               a(kk)=tran(1)-a(kk) & age(cc,rr)=mean(a(kk))
            endif
         endif
      endfor
   endfor
endif

; compute time since firn disappeared (for cells where firn is now gone)
ll=where(year_firn_lost ne noval and firn eq 0, cl)
if cl gt 0 then time_since_firn(ll)=tran(1)-year_firn_lost(ll)

; evaluate total firn thickness throughout time
; unsure how to meaningfully write out...
yrs=indgen(len)+tran(0)
for i=0,n_elements(yrout)-1 do begin
   ii=where(yrs eq yrout(i))
   o=dblarr(nc,nr) & for c=0,nc-1 do for r=0,nr-1 do o(c,r)=firntot(ii(0),c,r)
   WRITE_AGR,dirout+'firn'+string(yrout(i),fo='(i4)')+'_'+glacier(g)+'.grid',o,header=head
endfor


WRITE_AGR,dirout+'firnthick_'+glacier(g)+'.grid',firn,header=head
WRITE_AGR,dirout+'firnage_'+glacier(g)+'.grid',age,header=head
WRITE_AGR,dirout+'time_since_firn_'+glacier(g)+'.grid',time_since_firn,header=head


; write out statistcs
print, '**********************'
print, glacier(g)
ii=where(firn gt 0,ci) & jj=where(firn ge 0,cj)
print, 'Firn cover (%): '+string(ci*100./cj,fo='(i3)')
ii=where(age ge 0,ci)
if ci gt 0 then print, 'Avg age of firn (yr): '+string(mean(age(ii)),fo='(f5.1)')
ii=where(time_since_firn ge 0,ci)
if ci gt 0 then print, 'Avg time since firn loss (yr): '+string(mean(time_since_firn(ii)),fo='(f5.1)')

endfor

DISCONNECT_SERVER

end
