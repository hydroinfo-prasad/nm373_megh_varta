import os

def get_seq_data(path,inp_seq=3,pred_frame=1):
    files = []
    
    '''
    for file in os.listdir(path):
        if( file[-8:] =='TIR1.tif'):
            files.append(file)
    '''
    #print(path)
    files = os.listdir( path )
    files.sort()        
    files = [elem for elem in files if int(elem[18:20]) % 30 == 0] 

    inp_list, target_list = [],[]

    for i in range( len(files) -inp_seq-pred_frame + 1):
        inp,target = [],[]
        cur_date = files[i][6:15]
        prev_time = int(files[i][16:18])*60 + int( files[i][18:20] )
        #print(cur_date,prev_time)
        inp.append( os.path.join(path,files[i]) )
        flag = True
        for j in range(i+1,i+inp_seq+pred_frame):
            date = files[j][6:15]
            if date != cur_date :
                flag = False
                break
            time = int(files[j][16:18])*60 + int( files[j][18:20] )

            #print( i,date,cur_date,time,prev_time )


            if ( time - prev_time != 30):
                flag = False
                break

            if( j < i+inp_seq ):
                inp.append( os.path.join( path,files[j] ) )
            else:
                target.append( os.path.join( path,files[j]) )

            prev_time = time
        #print(j,i+inp_seq+pred_frame)    
        if( not flag ):
            i = j
        else:
            inp_list.append( inp )
            target_list.append( target )

    #print(len(inp_list), len(target_list))
    return inp_list, target_list

def test():
    path = './../pytorch-unet/INSAT3D_TIR1_India/'
    inp,target = get_seq_data(path)
    print(len(target))
    for elems in zip(inp,target):
        for file in elems[0]:
            print(file)
        print(elems[1])
    print("\n")
    