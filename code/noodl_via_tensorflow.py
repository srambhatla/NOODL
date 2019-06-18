import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import time

#rng = 21

# Set Parameters
n, m, k_max = 1000, 1500, 100
p = 2000
C = 1.
eta_x, tau, eta_a = 0.2, 0.1, 15
# Training batches
T, R = 50, 100
tol = 1e-7 

def nrmc_np(X):
    # Compute norms of columns
    x_n = np.linalg.norm(X, axis=0)

    # Check if norm is zero
    x_n = np.where(x_n==np.zeros(x_n.shape), np.ones(x_n.shape), x_n)    
    
    # Normalize columns
    X_n = np.divide(X,np.tile(x_n,[X.shape[0],1]))
    return X_n

def nrmc(X):
    # Compute norms of columns
    x_n = tf.norm(X, axis=0)

    # Check if norm is zero
    x_n = tf.where(tf.equal(x_n,tf.zeros_like(x_n)), tf.ones_like(x_n), x_n)    
    
    # Normalize columns
    X_n = tf.divide(X,x_n[None,:])
    return X_n

def HT(A, Y, C):
    AtY = tf.matmul(tf.transpose(A), Y)
    X = tf.multiply(AtY, tf.cast(tf.greater(tf.abs(AtY), C/2.), dtype=tf.float32))
    return X


def gen_coeff(m,pt):
    k = np.int64(np.random.uniform(low = 1, high = k_max+1, size=[pt]))
    X_o = np.zeros((m,pt)) 
    
    for iii in range(0,pt):
        
        idx_list = np.arange(m)
        np.random.shuffle(idx_list) 
       
        x_gen = np.zeros((m,1))
        x_gen[idx_list[1:k[iii]+1]] = 1     
        X_o[:,iii] = x_gen.reshape((-1))           
    X_o = np.multiply(X_o, 2*(np.random.normal(0., 1., [m,pt])>0.) - 1.) 
    return X_o.T


# Generate coefficients for the entire training process

X_all = tf.data.Dataset.from_tensor_slices(gen_coeff(m,p*(T)))
X_all = X_all.batch(p)
iter = X_all.make_one_shot_iterator()
X_batch = iter.get_next()

with tf.name_scope("data_gen"):

    # Dictionary generation
    A_o = nrmc(tf.convert_to_tensor(np.random.normal(0, 1, [n,m]), dtype=tf.float32, name="A_o"))
    
    # Initialize
    noise = nrmc(tf.convert_to_tensor(np.random.normal(0, 1, [n,m]), dtype=tf.float32))
    noise = (1.0/np.log(n))*noise
    A = A_o + noise
    A = nrmc(A)

    # Coefficient generation placeholder
    X_gen = tf.placeholder(tf.float32, shape=(m,p), name="X_gen")
    Y_gen = tf.matmul(A_o, X_gen)

# Run Tensor Flow
init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    A = A.eval()
    A_o = A_o.eval()

    # For each training batch (outer loop)
    for t in range(0, T):
       
        start_time = time.time() 
        AtA = np.matmul(np.transpose(A), A)
        print('Batch number '+str(t))

        # Generate data and feed
        X_o = X_batch.eval().T
        Y =  sess.run(Y_gen, feed_dict={X_gen: X_o})

               
        # Do HT (initialize)
        X = HT(A, Y, C)
        X = X.eval()
        
        # Coefficient Update
        for r in range(0, R):
            X = X - eta_x*(np.matmul(AtA, X) - np.matmul(A.T, Y))
            X[np.abs(X)<=tau] = 0.
           
        # Dictionary update
        gr = float(1/p) * np.matmul((np.matmul(A, X) - Y),np.transpose(np.sign(X)))
        A = A - eta_a*gr
        A = np.float32(nrmc_np(A)) 
 
        print('Error in Coeff = ', np.linalg.norm(X-X_o)/np.linalg.norm(X_o), ', Error in Dictionary = ', np.linalg.norm(A-A_o)/np.linalg.norm(A_o), ', Time taken = ', (time.time()-start_time))
       
 

sess.close()

#plt.figure("Tensor")
#plt.plot(12,12)
#plt.show()
