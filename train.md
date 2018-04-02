# Train the neural network

In this section, we will discuss how to train the previously defined network with data. We first import the libraries. The new ones are `mxnet.init` for more weight initialization methods, `matplotlib` for drawing, and `time` for benchmarking.

```{.python .input  n=1}
# Uncomment the following line if matplotlib is not installed.
# !pip install matplotlib

from mxnet import nd, gluon, init, autograd
from mxnet.gluon import nn
import matplotlib.pyplot as plt
from time import time
```

## Get data

The handwritten digit MNIST dataset is one of the most commonly used datasets in deep learning. But it is too simple to get a 99% accuracy. Here we use a similar but slightly more complicated dataset called FashionMNIST. The goal is no longer to classify numbers, but clothing types instead.

The dataset can be automatically downloaded through Gluon's `data.vision` module. The following code downloads the training dataset and shows the first example.

```{.python .input  n=2}
mnist_train = gluon.data.vision.FashionMNIST(train=True)
X, y = mnist_train[0]
('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Downloading /Users/markhama/.mxnet/datasets/fashion-mnist/train-images-idx3-ubyte.gz from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/train-images-idx3-ubyte.gz...\nDownloading /Users/markhama/.mxnet/datasets/fashion-mnist/train-labels-idx1-ubyte.gz from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/train-labels-idx1-ubyte.gz...\n"
 },
 {
  "data": {
   "text/plain": "('X shape: ', (28, 28, 1), 'X dtype', numpy.uint8, 'y:', 2)"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Each example in this dataset is a $28\times 28$ size grey image, which is presented as NDArray with the shape format of `(height, width, channel)`.  The label is a `numpy` scalar.

Next, we visualize the first six examples.

```{.python .input  n=3}
text_labels = [
    't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
]
X, y = mnist_train[0:6]
# plot images
_, figs = plt.subplots(1, X.shape[0], figsize=(15, 15))
for f,x,yi in zip(figs, X,y):
    # 3D->2D by removing the last channel dim
    f.imshow(x.reshape((28,28)).asnumpy())
    ax = f.axes
    ax.set_title(text_labels[int(yi)])
    ax.title.set_fontsize(20)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

```{.json .output n=3}
[
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAA1oAAACeCAYAAAArM3uhAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAIABJREFUeJztnXd8XcWZ/p+5Rbpqlmy5yF3u2BhMM5gOoQY2ISSEkEAI\nsCG9bJJN3/yATSEbshs2vQfYACEFQuihg+nVGGzc5V4k2ZbVrnTL/P44V6Bn5khXEle2kZ7v5+OP\n/J47M2fuOe+Zcu488xprLYQQQgghhBBCFI7Ivq6AEEIIIYQQQgw1NNESQgghhBBCiAKjiZYQQggh\nhBBCFBhNtIQQQgghhBCiwGiiJYQQQgghhBAFRhMtIYQQQgghhCgww2KiZYy5zhhjjTG13Y7V5o5d\nt88qJvY6uXv+SB/TXpJLf8kg1UU+OEwwxjxijOlXLI3++KoYvoT1b33IU2eMqRu0SolBR/dQiLcH\nw2KiJYToO+rA375o8r5/MdTuhyb/QohCY4y5Mte2nLSv6zIYxPZ1BYQQQoQyF0Dbvq6EGJKcsq8r\nIIQQwwFNtIQQYj/EWvv6vq6DGJpYa9fs6zoIIcRwYJ8vHey+tMIYc4Ax5u/GmJ3GmFZjzGJjzOlO\n+h5/YizUMg1jzHhjzM9yS6g6jTH1xphbjTGHO+m+ljvf53soZ4IxJm2Med45HjPGfMoY87QxZo8x\nps0Y85Ix5jPGmIiTtvv1mW2MucUYs8MYkx1qP7PmNFF/M8asNca0567NE8aYi3pI/0ju2sSMMd8w\nxqwyxnQYYzYaY/7LGFPUj3N/OXdNnzDGjOpD+knGmJ/m6tphjGk0xvzDGLOwP9+5W3l5fb9b2uKc\n7y3N+c4eY8zjxpjzeyn/fGPMY8aYpty1XWqM+boxprhbmpNyOqKpAKbmrm3Xv+sG8r2GA8aYdxtj\nHjTGbM35whZjzKPGmE+FpO2zr4Yt0+re/hljPmSMecYY05Jrq64EsC6X9CPO/btkEL666IW3ej+M\nMccbY+4wxmzK+cq2XJ9xRS95Pp57tpPGmO3GmF8bYypD0nnLg003Taox5sxc+9rU7ViXxvBE57tc\n2cdLIvqJCfiMMea13D3dnOt3wu5pj/fPSXeACcYTG00wvtlujLnJGDMnpMxxxpgfGmNW5Pql3bn/\nX2eMme7U8yPGmCdNMF5K5sq/zxjzgcG5OqIQGGOONMG4cnOundlqjPmnO57oyxiiW9qTc23PMhOM\nT9qNMa8aY64wxiSctHUAutq0h7u3LYP2pfcy+9MvWtMAPAVgKYBfARgP4AMA7jHGfMhae8veqIQx\nZhqAxQAmAHgIwM0AJgN4P4CzjTHvs9bemUv+fwC+C+BiAP8bUtxFAKIArutWfhzAHQDOALACwE0A\nkgBOBvATAEcB+HBIWTMAPANgJYAbAZQA2DPwb7pf8gsArwF4DMBWANUAzgLwf8aYOdbab/WQ7yYA\nxwO4B8E1OQvAVwCMBXBpbyc0wcT2WgCfBXArgAuttck8eQ4D8E8AowDcl8s3GsB7ACw2xpxrrb07\n77d9kz77fm5Afh+AEwG8DuBnAEoBnAfgFmPMIdbabzj1/R6ArwNoQHCtWgC8E8D3AJxhjDndWtsJ\noA7AVQD+LZf12m7FvNyP7zNsMMZ8DME924bguW5A4HcHI/C9nztZBuyrDl8CcFrunA8DqATwCIAq\nAJ8HsATA37ul1/3b+zyCAd4PY8yZAO5C4CP/ALAZQXszF8CnEDynLj9A0K/cgaB9OhnA5QBmAnhH\nP+p9HoAzEfjoLxG8eHk5d84rAKxHtz4NwfcUg8O1AD6HoD/8NYAUgHMQjBOKAHSG5Am7fwDe8Ktb\nAXSNQ1YDmATgvQjGNydba1/MpS0F8ASCscf9ufQmV945AP4KYG2u6O8i6GPWAfgzgCYE/dhCBGOn\nvTJ+E/3DGHM5gnFXBkE7swpBX3QEgnbmz7l0fR1DdPFVAAcAeBJBO5YAcCyAKwGcZIw51VqbyaW9\nFsHY6UQA1yMYhwwtrLX79B+AWgA29+8a57MjEDQsuwCMyB27Mpf2pF7Kus45fl3ueG0f0t6XO/5N\n5/gxANIAGgGUh6SfH1Kf1wB0AKjudqyr/j8BEO12PArgd7nPzunh+nxvX9+vQfaFGSHHigA8mPOD\nic5nj+SuywsARnU7XoagA8kAqHHyWACP5P6fAPC3bvcj4qS9JPfZJd2OxXJlJwGc6KSfgGBAtBVA\ncaF9P3f867n0dwOIdTs+FkEDZQEc0+340bljG7pfi9z3uCP32Tecc9cBqNvX/vB2+JfzvQ4AY0M+\nG10oX+12rKv9aAVwaC8+dd2+vjb6N/D70a1dWtCbX+Xs67o941O6HY8heGllARzp5PGe8W7tXRbA\nmT3Uy/NJ/Rs03zkmd71XO21GAsGLOdv9Hua7fwBG5vqTBgDznM/mIxg8v9jt2Lty5f0opKwiABXd\n7EYAmwCU5vNX/ds//gGYh2CMsRPAgSGfT8r9HcgYYjoAE1Lmt3PpP+Ac7+rXTtrX12Uw/u3zpYPd\naALwn90PWGufR/DrTRWAcwe7AsaYSQBOR+BQP3Dq8iSCX7dGIXj708X1ub8fcco6AoEj32Wtbcwd\niyD45WQbgC/YN2f0yP3/Swic7cKQ6m1H+FvMIYMN0Q3Y4C3JzxA81D0JuL9qrd3ZLU8rAr+JIJiw\neJhgeeADCPzqq9baz1prs32o5tkI3vD9xFr7qFPXLQj8pqaXuobRH9+/DIGPfNFam+6WfgeCRgwA\nPuqkB4DvWGu3dUufRuBvWSe96D9pBB0WYa1tCEnbb1/tgV9ba1/qb0XF245290APfgUA/2mt3dAt\nXRrAH3Lmkf045+3W2nv7kV4MDl2/cH/XaTOSCF649URP9+9iBP3JFdbaZd0/sNa+CuA3AA41xsxz\n8oX5YKe1ttk5nELwwshN25O/in3LJxGMq75trX3N/dBauyn3336PIay1a21uBuXwo9zfM95i3d9W\n7E9LB18MeXCB4E3wRwAcijcnNYPFobm/j1trvYETgqWEF+XS3ZA7dhuCgfKFxpivdZs8dU28ruuW\nfzaCidoqAP9hjAmrQzuC5SEuS6y1HX38Hm9LjDFTEPzkfAqAKQiWR3ZnYg9Znw85tjH3d2TIZ+MQ\nLImYDuAia+1N/ajm0bm/U3vQJszK/Z2L4FenvtAn3zfGVCBYBrTZhm+U8FDu76Hdjh3mfPYG1tqV\nxphNAKYZYyqttU19rK94kxsB/DeAZcaYPwF4FMAT1tr6HtL311d74tl+pBX7GcaYKry5RLc711pr\ndyPwq/cCeMYYcwuC5aFPdBv8hCHfGlp0td2Phny2GCGTmhw93b+uvmtBD33X7NzfuQCW5c67GcDX\ncsvl70bQb77c/SVxjhsRvEReZoz5cy7vU+pT9msW5f7ekyddv8cQxpgyBEumz0XgVxUIlp120dNY\nbkiyP020tvdwvGsG7Yk/B4Guc2zt4fOu41VdB6y17bmG5XIEv4bdk9PRfBBAPdiJq3N/Z+FN8V8Y\n5SHHtoUcGzLkhLXPIhgQPI5AY9CEoDOpRTDh8ESXAJAbmLh0/doTDfmsBsAIBEsdFvezql338P15\n0oXdw57oq+/32z/7mGdKLo86xX5irf0fY0wDgvXsn0MweLbGmEcBfDn3y2T39P311Z4Y0u3BMKAK\n4X3AdQB2W2tvNcb8C4I3xpcB+DgAGGNeAPB1a+39IXnlW0OLrrbb6x+stelcuxNGT/evq++6PM95\ny3Pn2GOMWYRgJc278eavEA3GmJ8j+IWj64X0FxDotS4F8LXcv7Qx5m4AX7LWrs5zTrH36RonbM6T\nrl9jiNw+BA8h+BX9VQT6vHq8uerjCvQwlhuq7E9LB8f1cLwm97drENi1vCtsklgVcqw/dJ2jpofP\nxzvpunCXD56NoFG7yfllrCvfbdZa08u/aSHnHjI7sPTAFxFcs3+11p5krf2ctfZb1torEejgCskS\nBPdqIoDHuu+e1Ae67uE5ee5hf5Z59tX3B+KfA/Vp0UestTdYaxch8N+zEWgtTwBwnzFmzGCddpDK\nFXsBa21dD+1GXbc0d1lr34Hg5dMpCJbdHAjgzpDlXQWt3iCWLfpOV5vs9Q/GmBiCDZjC6On+dZW3\nIE/f9cbKIWvtJmvtvyLQAM9H8DKpEcD/y/3rSpex1l5rrV2Qq+/7EKz2eTeAe8N2phP7nK4XM/l+\nXervGOIcBJOs66y1B1lrP2at/WZuLPergVb27cz+NNE6LLc0yuWk3N8uPcKu3N/JIWn7o3EIo+sc\nx+UaMpeTc39f7H7QWvsEguWA5+S2Xe2acLlLHV9H4NyLcrN+8SYzc3//FvLZiYU+mbX2jwAuQLCB\nxWPGmNl5snTxdO7v8QWsTp98P7e8cA2AicaYWSHpw/yzy6dPgoMxZiaCHafWOb+0ZNC/N+ACwa9V\n1tq7rbWXI/hlYhSCCdfepGtJj+7f/sFbvh/W2lZr7UPW2i8i2OWrCMGOX/uCLORbe4uudjys/zsO\n/b8PA+67bMBr1tqfINjtFAh2igtLu8Nae6u19nwEv2zMQDBJE/sXXf6Qry3p7xiiayx3a0hZPY3l\nhnS/tT9NtCrR7Q0J8MaGEhcimCnfljvctf740u6TIWPMZDd/f8mtf78fwVI1Wj9vjDkKwIcQTPRu\n8zIHk6oEgiVEZwF4xRWr58SDP0HwBuDHxhhXg9QVw2sw31bur9Tl/p7U/aAx5gwM0mYN1tq/ItgK\ndzSAR40xB/Yh2+0IJjufNsacFZbAGHN0bmvcvtJX3weA3yNY63yNMSbaLf1oAN/qlgbO//+j+68r\nubw/RNAG/M6pTyOAMWH+KZhcvJAwseXY3N+2vVkfBO2TRbCUQ+x7BnQ/jDEn9PCyr+vXjb3tV100\nIvwlpyg81+X+ftN0i+2Yi0N09QDK+wOCF71XGGO8zVGMMRHTLTanMeZAY0zYagvyQRPEdTw2pLw4\ngpdNb6TNHe+K93XdAL6DKBy/QLC0+FthY87c5nBA/8cQdbm/JznlTQfwXz3UpTH3d0j2W/uTRusx\nAB/NTWiewJuxhCIAPm6t3QMA1tpnjDGPIXhT/Kwx5iEED/67ECwxe6udwCdy57/GBAFjn8ebcbSy\nAC7tYeOC/0Owc9xVCGJU9LRxx7cBLMid5125+m9GMDCbhSDWwDcRiFGHEz9HsL77L8aYvwLYguAt\n2JkIYjkMStBDa+0/jDHnIJjMPJKL77Ckl/QpY8x7EfjaXcaYJxHEmGlD4CcLEWyyMR59Hwz1yfdz\n/BDBG6hzACzJrYEvReCfYwH8wFr7hu7MWvukMeYHCGI1vZq7tq25MuYj0Khd49Tnwdz3uDf3rHUg\n2Izljj5+n+HEbQBajDFPI+hgDII3xgsRbOX+wN6sjLW2xRjzDIDjjTE3Ioi7lwHwD2vtK3uzLuIt\n3Y8fI/jl+gkEftUJ4HAE8bDWA/jToFa8Zx4EcIEx5g4Ev7ikADxmrX1sH9VnyGKtfcIY8xMEm0x0\ntd1dcbR2oWfNTE/lNRpjzkPQZj1tjHkQQQgai6DvOhrB8ueugLKnIRgHPYXAb3cg+PXiHARjoa5+\nowRB/MjVCNq89bkyTkOwscY/rLXLu1Wl6wV/GmKfYa1dZoz5FIJYay8ZY25HsDKrGkH/tQfAyQMY\nQ3TFZ/uiMeYgBL+ITQHwLwhiaoVNph5G4FNXG2PmI7dyzVr7ncJ+632E3fd7+dciF2cEwUN5O4KL\n3IZg0HlGSJ4qBFuR7kAwCHwVwMdQgDhauc8mIpjtr0fQwTUgCDa5MM93eSBXZgrAuF7SGQRBiR9E\nEMOgE8FkazGAbwCY3Jd6DrV/COKGPJS7/8256/EeBG9GLIArnfSPBC4cWtYlcGJg5Y6HxoHJnaM5\ndz8W9lZG7rOxAL6f8702BDFIViEI4ngRusW4KqTv5/Ilcn7yKoJdKruu1Qd7OdcFuTTNCGKAvYZg\nQp8ISVuW8/9NCDrDYeF/A/TZTyAYuKzN3bedCDqWr4DjzBTEV9GHeCMIlm7cgeAtYbYnH9a/veYj\n/b4fAM5HEE5kVa5t2ZN73r8LYIyT9jo4/Vu3z3pqO+vQcxytHuuWa/duQrBBQyasbP0rqO8YAJ8B\nsBzBWGcLgnAnle497Mv9y6WrBfDTnG8lc771OoKXxe/plm4ugP9B8LK5Pnf+OgR9XPdYjfFce3cP\ngtA4yVz6p3PtY5Fz/h/l6nnqvr6++vdGnKy/IRhPd+Z87F4A5znp+jOGmIxgJ8rNCMYor+V8JIae\nx2AXIXhp3Z5LY/f1tSnUP5P7gvsMY0wtgmji11trL9mnlRFCCCGEEIOCMeZFAClr7VH7ui5C7A32\np6WDQgghhBBiCJLbLGwBgl0JhRgWaKIlhBBCCCEGFRsEtB2SO8sJ0RP7066DQgghhBBCCDEk2Oca\nLSGEEEIIIYQYaugXLSGEEEIIIYQoMP3SaBWZYptA2WDVRexDkmhFp+0IC7xaEOQ7Q5tm7Gqw1o7J\nn3Jg7DP/cWMRD2AFQKqG622yfhrrvPKyzmnD8mSL2C7e2NrvuhXi+71V1Pa8SecErmeiooPsZHOx\nlyfewvcskuTwROnyuJfH9R1TniHbdYOiTb5f2I4O79i+YMi2PYNBeUgM+pZ2Mk0R+0umzHEWAJFd\nA2hr9kP2y7bHbZMHhPu8+mW6R/bb1W0hl8MY7jALUvcBlNHXtqdfE60EynCUOaXflRH7P8/YBwe1\nfPnO0OYB+9f1g1n+vvIfU8wD24EMLjdddgzZ8RY/TcYZ/2ScsXG008/TOoUHx7M++0zvFYn4GnQT\n5WM25ZworNMvcIe8X7Y97vd2OnZk+doXivWfPJrs2SeuI3vVw9O9PDVP8T0rXbGD7J1HT/DyNE/h\n7xM/ZifZnWn2i6lfT3plZFau8Y7lZRAm9kO17RkMsoce4h2LLH6Z7FjNJLJ3L5ro5Sn/S5625m3C\n/tj2uH0Oss4zEunDRCzjtE/RkLbfeRaznSnnvIPTxuXF6adMyPc1RTz5t6k+xL7Oc90G0rf3te3R\n0kEhhBBCCCGEKDDa3l0IIXpgIG+5Vv/xULJ/tuh3ZC9L+m+I/7KR88yqqif7izX3e3l2ZMrJvure\nd5FdfuZazhDyhtLuq7eW+zveLy0hazfzEJs2leyKPzZ7aZbVjyM78RC/dV26bAqXObvNK6PlCD52\n76F/I/u0V8/38rSu5dUuIw1/3/ZNFWSv+ugIr4xMYjTZEx7lz8v+FvKrh3td94Olq28XYtNrvWNl\nN7BPxZw1xo3H7iJ7/dn+0sGx/8G/kv7mgD+S/d7r/t3LU/4XtqufGEn2s0/NIXvGvz/tlSF6IN8v\nWO7nIWki5bxcMbPHX0aRr+2PTeZfNr/66J1emsufu5jsVAP717Q5W7080ysayd6wyGnTnHrZsKX2\naf4FKzqC26dsX/rtsOs4SOgXLSGEEEIIIYQoMJpoCSGEEEIIIUSB0URLCCGEEEIIIQqMJlpCCCGE\nEEIIUWC0GYYQQvRA9kTepKLu476A9qMHPUH2qeYxsn+/7TiyJ5Q0eWVs28xi8l3NpWTfVz7fy/PQ\nDhacLxyzgewL1r5A9gf/+UmvjFk3sGjYPLmEEwzXzQncrfAdgbYrvgaAtb/lzS++eBBvHT0xzluo\nA8CSCs7zm4bjOc89XI/6Q/zNDHbWshD+qSRvDz2lgjdEAID5R7BIPWb4+71QNJnsz0/3t8G+8voL\nyW69mL/fe6/a4+W5b75z3Yarf/WB6Ohqsj9w92IvzYgob7u/tmMs2Td89kyyJz/gbxLw5fffRfaK\nFJfROcLfjWD7ZzlkxYyiF8m+9bxryT6n7HNeGbM/+ax3TPSBkN0hTJS3O8/s9vsYl+jMaWSvu2g8\n2eVHNJD96SUf8sr45Hzu62YVbyP7/ia/3/rH0gVkV9/BG7rsWs5+P+cX/oYa6bV1ZGf2+G2Ni4k7\n8eD6sk1+gdAvWkIIIYQQQghRYDTREkIIIYQQQogCo4mWEEIIIYQQQhSYva7RctdJ2lTnoJyn/Zwj\n+bzOstbyl7d4eWxpgvN0cN2S0znIIwCsfT+voZ9yN3+euEPrkIV4u7Du+0eT/Y33cPDXZDbu5WlI\nc3DXHR1VZNckeP348RUrvTIOO66O7OYsa3EmxznIIwAsLeHAx0t3TSC7PcPasE8c+7BXRvMibvNu\nfGYR2bM//pyXZ1gEmc0TzHP11w70jn1+Pgf0/OOGo8huT/m+0/oMB/1973s4sOvtu7gMO541OQCQ\n7eA+6MrP/it/XuRrEZoncNe/+2AOAFqxkj+/5h1neGW0T+Y87fUcQHvFeA7GDAB1t8wiu/YDr3hp\nRMD6j7EG8/CEHzD2hl3cXiUiKbKv/NwNZEdCAm9vTrE+NOoMlq7+l5u9PCMi7IcPN88l+7amw8i+\n4uS/e2XcjAneMZEfU1zsHcu2tpIdHcNj1c0X8nMHAHvm8/h23IQdZG/fxv1YrMFvv37ReQLZoys5\nMPKWDay3AgCT5N93Gtr4PKXTub9c9hXWDAIAovz9pt3CPht/gPXJgD/XMLG9N/3RL1pCCCGEEEII\nUWA00RJCCCGEEEKIAqOJlhBCCCGEEEIUmL2u0RqIJqv+k7wO+ZTLn/bSnDbiVbKTlnUQ7y5rI3vO\n7/yYMuOe53X5G9/Jn69796+9PC84Oq41p/Da0fN/xfEMpv39Y14Zsz8lHZcQ+wOXnPUQ2UtbJ5Hd\nEaLRKotxbJrWNK+hH1fMa86XJ31twpGla8g+raiO7G9vP9XLUxVvJ3ti9W6yGzpYM7O0mTVdALC8\nkde/f/DIZ8h+edZML09m1Vrv2FAnNr2W7NTotJfmxqvO5jSlrI0a+Tr3QQAQn8raggf+yBq5ckcq\n1hIia4nVs0/Ov4pjGt31ykFenhLnFlY/xzqvDkda0fIca8kAYPQG1uY1sJwM6awTiwzAR+ayfz2e\nYH1QNulr0IYrxYtYlxmBr4WckWBdTcryNb+9keMAbm9nPSkALBy1nuyM5ffvz++c4uUZUcT3aW4F\nx0+KO3HZauMckwkA7LFnkW2eeNlLMyxx4jsZRxPr6rHC2PpbfoCb63xtXtzRXO1o5We89h5nPHyK\nf55sXRkfeN6J83emr3UtbmQfjbOsCy0Z7reiHb7GNN7MPtr4GY4VWJXgPRoAIHEnj7NdjZZN+216\nodAvWkIIIYQQQghRYDTREkIIIYQQQogCo4mWEEIIIYQQQhSYva7Rcskef6h37M4/sRZqSSfHcikz\n/lrKVSnWRm1LVZL9y928FvznH/L1Vp/t+DjZc2bz2uXvNBzg5RkZ4/WyE+O8VvQxZ8n5y+/+X6+M\n8nNY0/HO8y8jW2uX9zPcOEJhFCC20M7LWJs49v6NXpr0xk18oC8xjoZDHKQ+sONTx3jHxsVvJbsx\nxWvQp5fUe3n+seVgsieUsS4zHuF16g0pXoMOAFvS3D7VOe1ZXcsoL08my+/JyuKsFat0dBSjivy1\n/ZNHcF1bMtwWrfpXPxbS9K8NP41W49E1ZJ+6YKmX5on1C8hOzuL7UXXRTi9PTZzvUWeWu+T639eS\nPX2y87wDePfhSziPE9dt6mRfH5OdxG3Aj2bfQvbLSV+X4/K9+95D9pzZm8k+oHyrlyflfL9tl3G8\npbE/fzLveYcLX5rzANlt1h+uHVzMfcJz7dPJnlLCPje6yBHEAFjaxMK/rKPROrDSv48Z5x39wSVc\nDzcWVzxkzLb12FKyJzzhJRmeZBxdU1FReLpubPl37sti4Ptu475GC2O5fSp/nvu6VBmPC7JF/jjB\nlnJdGw9yfdTXaDnuhfLNXLd0gjVcydlhsQO5nxrl7MHQ+Rn+bgAAJwyd3YvjHv2iJYQQQgghhBAF\nRhMtIYQQQgghhCgwmmgJIYQQQgghRIHRREsIIYQQQgghCkzhN8Pop9D+Pb96wDt2uxM4bX0n24kQ\nYeXkIg7uN8HZlKI+PYLsbekqr4yln/wp2c91cN1XdvrC8CInMF9jmoXubvC/e1IsegeAwxIscL7n\nz78n+6yJLBgORRscDB576dpGR3OQwRM+zcE9l6w9xM/jbobRl7rJNwAAuw9JeccSET5WEmV7apG/\nscDY0may96QSZLc4AYxHuxEa4bcTbmDkk8es8PK4aRY3zCB7ZQMHI144YYNXhkvEsG8k5jT1kHJ4\nUXkZP2cHl/ubUow5j/3gmcZasr889V4vz80NHKDYDUq7/ly+/i+tmuqVsb2NN79YUL2F7BPHrfLy\nPLeTy/nmunPJXrGaN0gwKf+d7NQDeZOEyWUcMPvWjX57ZS23peXbfLG8CDivnIMAr07512pbhsc1\n04u3kz0mxsHS3fYNAPakub0qjvD46piK1V6eqihvrPNYC28UlnTapvOrOFgsAKQXNnvHBADDz1q2\nzQ907nLiB14gu6GTN7ZYG/N9p6GR242Wg3gDiZZanh4U1/ttQEeU26uIG1w47W8c1jmS67LtOP7c\nFrOPFiX88X5nDdelPcX+NqmC2yIAcL3NdoRsmDFI6BctIYQQQgghhCgwmmgJIYQQQgghRIHRREsI\nIYQQQgghCkz/NVrd9SphWo88+o/VP+I16ceXXuul+cceXts9v8RfD+/yavskssfGeW1yynIQtJ1O\nIFIA+K9GRwvmrGd2dV8AsLaDdRCTHK3YFkeT5a6hBoA7mw8i+9TyZWSv/qMf1HnmRS/xAelu+kYk\n6h+zIcH8+ksBdFwbf8MawKIWfjwbv+Cv1Z6weiLZ6U0cNLQvwZVN1Lkmzhpxm/bX9Q8FfxtZs8c7\n5rYTWUdT8s/dB3p50k7g4Cll3E5sTXLw9Ee2zPTKGFfOuq2jRtWR/Vi9n8etW0mM71M8ymvhW9N+\n4Mv2NK9t35HkdfvTRvlBdvfeyvb9B/favtIyyUtzweinya6MtZN99bqzvDwHjWQ91V2LDyf7Z//y\nB7I7rd9+ff7hC8l2NVr/98jxXp5Z87lPXbmB256SjewX5Rv9573swE6yzxzJQZyf3eoHPV40oY7s\nS3/IgZKvuJW//3AiUsHP3qPtHNB3Rsj44/N/v4Ts28/7EdmbnYCxScv3FQDmlrHWLmHY1zPw+5Cj\nilmj9bWMy5ZiAAAgAElEQVTl3C42bOU279NnLPbKOKV2Jdm+knB4YqJO/xvS/bqcXsXP3r/dfTGX\nUeJrtMZMZB2Tq9kqrue2ZvKD/vhj/VklZLdPZj1VvNGfYkx6iNuNhoNZw9yygL9wKumXES/lNDvW\nsr79xGN9XeGr4znofHrrNi/NYKFftIQQQgghhBCiwGiiJYQQQgghhBAFRhMtIYQQQgghhCgw/ddo\ndddmhOldsr3HxbjtXNZkvR4Sm2p0jHe8d/VXxSGxIMqjSbLdGDM706zJGh33YzhknfhcEcPaHTcW\nFwDEnZgTbVleb+qW8XzrdK+MXWlei/1ijNe2r3kHr9MHgLOq30F2ptHRUrj3Js99GTYM5Dq4Oqe+\n+H0frv+67x9N9gGj1pG9bAuvKb5w3nNeGc9Usr4PjpzRFPm6HJe9GU9if2LR+PXeMTf+ixvzyo0x\nAwAjYtz2tGa4DdjZwc/3rFF+LK7DK7kubU4Z40t9PVndnlFctyjXbVZ1PdllMV4bDwCdGSdWilPG\n2ITfTi73jgw9zOGsOUlmWM+wtpk1AQCwrJz7KbfP2d7EGggA2PAaa5LKNvK7zx+tP43s1VvGeGVU\nLWGfbZjH5/3MKf/08tz+9VPJNmdyG1fsyIFKt/t+v/LJWrI3nsOe8ZW5/nldDeQDLb7mcbjSuXA2\n2WWRR8mujPhaqVk3sF+uOYf9sjbO44KXk7620NWeR51Ybo0Zjg0KAClH1zyunNuJXbu5HokQrXDY\nOE4AiPT++0f2RF+zv7qD+wcb5/sTbfKH+vXgWLLVz3Ka1jP4nq6a5o8lSkfweTN13MZlp3LfCADr\n3sftVdEo7mOLXmd/MyFDttKF3IdGKlgPOzvh66/ue/8xZI/7sTRaQgghhBBCCPG2RRMtIYQQQggh\nhCgwmmgJIYQQQgghRIHRREsIIYQQQgghCkz/N8PohhfoFIB1RP8NH2PB/7b0a2Sv7+QgwQAwxtkM\nY1eKxeQTi1kACvji8RbHnlLMgYSbswmvjKztfd7pbmwBAAnT+2YYbrC/SUV+AFCX5gwHgXvM1xOi\n7WbemKP4dKdcbX5ROJwAvmHX1sT4UbJp9ovdH+bnAAB+fN7vyf7MMx8iO+MELP7TSj+Y55TXlnrH\nqB4D2Oii9X1HkV35wlYvTbpuQ7/L3d+YXeqLYeOO8nZ68Q6y3QDlADC+iNujVe28wc+oYg702BYS\nOLghxSJid7OekqgvHK9KsAB4d5LbjZZObovcAMYAUFvB7UaRs9nH2JBNg5ajxDs21Kg7h9vX7HZ+\nFlOtfuDX+2LzyD6/hjev+cvmRV6eqhXcP0Q7eCOC9PfYl6ZG/U0Fth/B9so/zyF7aRXbAGA+wiL2\n7x98B9kjzuBO5z++f5lfhhPD+NNVa8i+sXm8l6c+zX6ecfpcV+gfefQlr4yhSnI0+1SZEzh4dJQ3\nOQGA7Cuvk73T2bjigCLeECcMb9zjjHPcDUwAoDTCdV2xhf107AvsHPELQjbyKNlO9qvwN3oZlmT9\ncWZ3thzjt7+jY/w8X34cb6Ty17pDvDxtSe6Hdh7E/jW2jPuXZKN/3rYmPhafyH1dPO73OUmn3YjF\nOE12Hvc5mTX+ZiyHjd1M9klVznOQ9vM0T3M2cPFSDB76RUsIIYQQQgghCowmWkIIIYQQQghRYDTR\nEkIIIYQQQogC85Y0WjblB8B0+dQXbiO7zbJuoCLiC5DWd7BuqzzGOhM3ECQATCvmtcjHlLG2YkfG\nCRbpx19ETbyJ7Sjbe0J0XWURrluro9Fyv68bfBkAKmPOulZH97WkfaqX51szeE39f897H9mZZSvJ\nNnFfF9KX+zfkCAmcSEG4w9L0Qe/marI6z2DhxNf/3/95eb681LlnSV4LH9vFj+f7jnrZK+O8dc+T\nfe7DnyZ73hW+DqnxBPbBjip+33Lghcs4/WlDM7CkG6gT8Nd2lxp+vhtS/tpvV7vpahpGF3NAxt0R\nf637mhZu804Zz1rW1W2+NmxUcSvZdTs5gPH4Sl63n7W+71c4wZbbM9xOuAHZASA6ZybZmRWrvTRv\nd2b8goOHv/7lWrIvP/VhL88RpWvJ/uWWk8geP4f7JAD4w3u4XfjANV8me/1ZrIW57DT/vH9Zx7qm\n1jbucz578CNeHtdnq6Pso6s6OFh6+fm+TvMrtVyXq+pZB3LLMl9TmqnnPtSNWTuzlevhtMxDmo4K\nfj6nOtqVhgw/72FML2Ifa8v2f4jnarbCNFoR5x19fDnr6Cv+9CTZmWv8O+mOt2ITF5Cd3rwlf2WH\nIe0T/fFIU4bHxGdUsHY7NdW/h3dt4mDhCxZx+/XAy6w5HfuEX8aueXxs6pGsu1u9mtsRAChfxW1a\nx2huE846jbWtjyeme2W0Z7iMqig/G1tSHIwZAMy4kM0O9hL6RUsIIYQQQgghCowmWkIIIYQQQghR\nYDTREkIIIYQQQogC0/8FvN31K662BUBsMus/xsRYK7S209cauHQ464pHx3nd9sxiX3dye+NhZH9n\nybs4QZbXP59+uB9/6P7lc8mOJxzdzW5e+w4AkTZeo1pay7qIEydxbJGTK5d7ZSxPTiB7TDHHEQiL\nCZBwYmx0/JjXn8ZO5fTDUo8VRojPepqssDT5WHQwmV//2fVkf2HJ+V6W9lb2p6ijySqfyxqiQ0vX\ne2Xc3cznvfrYv5H9jic3eXn+2HQQ2X/fzGvjn143jewZrUMzls0xic3esTtbOOZQtbP2uzhEs+TG\n3nLTPLZtBtnHjeO18ADwchu3m480c1vUmvbbnoYkr8ufN5bbxdGOhmtTm79uvTrurG132l43PiEA\npKv5vCGqx7c96a18LWd+ke1HQ2KJLZ79brIzK7ntf/9rq7w8dY6WoPgs1th8dcaDZLt9BQBcNe8f\nZLt9w5LkFC/PP7ex/uI9Tsir3/6U+8+2iX6buH0i1/25Q7gvnA5fU5qP4aTJcmkfx0+SG6vqy1uP\nCcnF9/qwIh4HPN/B2qmoG/wMQGeIBisfWbDGr6O699hPYTHAxkR5rNR6yESyi4epRstmer+WlZOb\nvGNtWdbWurq6cytf9PJc/+jxZC9uY61UtJz7scaD/elCdixrmDc2cpsQKfX7y0yC/TpVxf3nlvZK\n/vxhP9bukwt4TPyZGm4nm9Ls9wAwfyLrTNu9FIOHftESQgghhBBCiAKjiZYQQgghhBBCFBhNtIQQ\nQgghhBCiwPRfo5VHv7Lys5PJjjrxOloyvA60NOJrhypjvHqyKcPr4d016ADw5EbWlVS94qwDdcJo\nNc731wzbdr4cRa9yXbPV/nfPjOc1quk0r43d0DaS7MRI//tGnJXpuzO8vtRdfwsAr3WwpuNBZ53+\nWdXv4Ho27vTKYL2d//E+x9FOmShfWzd2VVge119NzHd5t5xIBTtLtpk1c7FaX/Pw5Rv/yPby88hu\nb/G1LrEtfCwxdzfZVx/IMeieaWWtDwDsSbOPLmthvdWKpCO+ALB0D+s8NtbxGuiaKY6vHMmaLgDA\ns77G8e3GprSvs6mKujHtuP16bpcf027hSNbOVURZJ3FwNWsNwuJZTSrje39kOeu4NiV9fVUswnVr\ndDRbezrZN8rj3FYBwOIG9qlZIzgeodt+A0DHKMdvvRTDE1eT5bKyzY8p879PnkZ2LUss8asvnUD2\nWeNf9cr41mvnkO36V+xB33d2z+M27zXnvidP5jZv9F/8/vKeYw50juTX1Ljtr8067XOE6x7axg9R\nwuIjdeeu1+d7x2aC9bMlhscKr3dw+18a8duAfISN0V7q4Hf0V591M9m/+zcej6Ws/90qnCBqTVN5\nzJZfzT80yaenP3GSH7cwLLZsd8aEaIsrp7LWq+ivPFbdcSzfs8kHh8TkbOGxqqs7T5T63yU9n8ut\nSLAfvPj0LP48xGV/f9x1ZLsxbsP62GOr+bo9gAovzWChX7SEEEIIIYQQosBooiWEEEIIIYQQBUYT\nLSGEEEIIIYQoMJpoCSGEEEIIIUSB6f9mGHm49r1/ILvRCbbb7GyG4W4EAfjB1lzqUmO8Y3PGcqDH\n5SewGC6d4jIr4yxYB4BxziYA7TUszqyK+YLOGSMb+DxZnrtOLeUy3esBAKPjLDzuyPJ5w8So7qYi\nTye5bq//qJbsWReHbIYxkMC8exOnfn0SRg/gO5k4C4jdzS+i41iWe8IdftDpH28+heyGzRx0L97o\nP2ozj+ZNFD43mYPuLWnnTTfCnouaYha0Zpx3J4eW1nl5bl5xONmRVi53RmUj2S+8c5xXxpRnvUP7\nPe59nFvkP1fbMizMrc+wyHhMgoOnA8C4ON+DhjSLbN0AxvnaNwCIm7Rj+5tSuHXZ2jaC7I4M+1xp\nzP++7uYXJU5bMz7Om3QAQPMkrv+w3AzD3XQHyNv23Ldirnfs2AM5iPGyFznNqO9Vk/3zi0/yyjh1\n3utkP+FsDNUxKSxIO5sTivk+/+KwG8m+tPkyr4hskjeTqXTa0TBRv804fajbxvceq3VIUz6RA/hG\nnLbcbPc3VKr79tFkZ/EC2W1ZzjMq5rdfKcvtRMbZSCBh/Pv4XPt0sj9RxZv3/PL095P9vQa/jEtH\nciey+whOM1w3w8jHIWUbvGPPNvP9qHDa8dKI3+ecPfU1sm8fzwGMEeeHcVSCN4oCgJQz3k22cRsQ\nCxkzjyrnchqbuY9NNHCZTQf4ZZxUwnX7XRN7Szzi55la1OAc0WYYQgghhBBCCPG2RRMtIYQQQggh\nhCgwmmgJIYQQQgghRIF5Sxote+wh3rEoWL+y0gmYOqWY9R+uHgkAJsR2ke0G2ctaf374b5PuJzsz\nidcZ16dH9GoDwLuqXya7Jsrai8asHxRut6PhcNc3FxleK5qI+MGWyyyvp90NDgK3M+Ofd0yMdUTu\nmulVp/yW7LNwmFfGXqebrsENPgz4wSvdBftewGI3PfIHvOyLzqv1vKPI/uB/3k32oztne3leWl5L\ndmIL+/XCM/1Aox8Z+wTZD+7hAKDlUfb7MK3eunbWK55SuYzsm3Ys8vLEn+O1yakJfJ2f3cDasGiI\nHOXtSPIgDqZ+X0gA2aijGa2KcPD0KSW+1tENULyqnTVtd6/gQKMfnv9M3rrWdfJ9rQjRlD6zg4Mn\nN7ezWuqkyRygcVvSX5N+ZAVrK15s4TJdzR8AT98zLBmAFvTwWl9b8c5qDvy9+wOse2r8lRMgu9PX\nHpTFuJ04ahJrPz962KNenmMTfF83pVm784MdJ5M9rsbX6k0b4eiPXf1VCH0KOj9MOXfaK2THjXOt\nxvqRWy87ZDHZL3VyWz46xrqvsLFTIdie4Xay8VOtZC8o9X0/6YyVTp/PmqG6wlRtyDGveLN37PHd\nPCbJOI10ZcRX0v5z8wFkt8zgsek5h/B4eHWzvzfCyIRz32O8B0FZsT9mqSljnxxRzH3b6gWOj6b8\nseK6FLdXbj9VbPx2xdUi7k30i5YQQgghhBBCFBhNtIQQQgghhBCiwGiiJYQQQgghhBAF5i0tWtzw\neX9Ntrs2NOusw92VZr1RmO5ke4pjEFVGed/9rakqL8+TqZlkjy1iDZMbr2tXmnVQALChYxTZ7RmO\nCTAixutRAX+//nJHr+Fq0CqjfhmRkBg53Qm7Rs2ZEsfmNbgvOGv5N37zGK+Myd99stfzFpxuuoaB\nrM3vS558cViyJx7qHdv6eb6+/z7vNrJ/tY7jS2zfwf4JAEX1/CjNPYXj43y+5gEvz592sRZsdJzX\nHTel+R5HjK8LOXYEn2eFo4l8/mlfT5adyr5RNtF5VpzzzDl1hVdG05Xeof2eNicunqvHCsONX+Xe\nE8DXZbpxs06cwfcoLI6Wm6fKafPixm9r543cTvbaGMdc2tnJbdy63dy+AQCcZfclUV6n/2rrRC9L\ncrREWgPB7QsB4DfruW0pjXNb1D6G34WOGMe6YQB4T9WLZF927+VkLx45w8uTbnO6/jSfZ8zT7KP1\nx/va4rFlTkymbH6NVpiuVgQ8exw/n2c0swb+gEkcKxQAvvgMx1C7uZnbf1cj3tmHGH59YYyj/Xq+\ng/Wu/3vQLWRfPePgkFJmOrY/NhJApJTb8QrjP4sdWX6eq5z+5DFf4ou2x7nxH7GI90a4fQn738jn\n/P0UZl/EY4NIlPvL0rhf11c2OX3KOv5+bnzR9f+s9cpYdsxosqujztgpk7+f3pvoFy0hhBBCCCGE\nKDCaaAkhhBBCCCFEgdFESwghhBBCCCEKjCZaQgghhBBCCFFg3tJmGDcu/J137Mm2WWS7wnp3Ywd3\n84ww3EBjyZAgx24gPjfNpCIOrjgu7ouK27LFvdYtTMTeluE8rmh9Vyb/5h+jnA0y3LqHBQ11N7+I\nOqL9uhSLBX9+2S+9Mq7+bphAde8QHTnSP1jE39u28XWxSQ7YGB3L3xEAdp7IwXbtRQ1knz/FD975\nbFMt2Vc99S6yIzFnh40Ql+2s5vt+Qc2zZC/tmOTlqXQ2V8k4PuwG964J8dk7Glmw+tArc7mq1SEB\nsiv5vJ2d/HzZNeyzM9/JgSQB4MVDD3IO/NVLs7+RjfGNO6PUF5c/muSNdlxRcWfWbzLdIORuO+EG\nOc6GOJDrC3En4OKaFt/Xq4t5w4zqBAcJTTv+tLPeD9K+aSoL8N3Nek6oYLE9ANwfWegdE/lZ/Rd/\nY5qSem5b1p/mbPgzj23/DgKf+OMnyI6Ucp9bstbf+Kl4J6cp28bn2TnX8dGU3wcte3Ya2TPj7Oc2\n5fd1XkD5PJsXDSeyzc29ft4xc5x3LOKNUbh9cjfDGAjRkE2YXHak2TPPHsF9lbuhAwBk29q8Y8LH\nTOXNIxIhm6d1ZqNOGvaLPzUu8vLEnMvvbhZRtNkZj0X9fsvd4CedYv8rifnjj8gq9oUxL/P3WTeL\nN3XKlvn+98N1Z5D90SkcuDts07mykLH33kK/aAkhhBBCCCFEgdFESwghhBBCCCEKjCZaQgghhBBC\nCFFg+qXRshWlSB91+Bv24cUve2nuaeZAYe0ZXueZjLBdEfUjqSWdaiWddcdh6y/dclwN05rkWLLd\n4LAAMDLW6h3Lh1sXV58xtogD+7kBmwGgNMLaI7eup4141cvzfHo62W6g5C0p1kC5+jMAiE2vfeP/\nZlOR93lBKS9B9rA3gwXfd8sfvCQXrHsH2VnLdWpLl5N9cOUmr4ziSB3Zz+2aSvZPXzjJy2M7+J6Z\nBK9rt30IdGeynOb3m44j+4IJz3l5ZhZzwFnXD55t5UCjP1p6ildGqsEJzOfoyWzWr3tLPftgbLez\nrrqh9wC8ABBJ7rv1zgMlzU0CMiEBi1ud5yTp3PuzR/pt3q0NR5DdnOYy5lZsI9sNng4AOzoryHaf\n59KQte6taX4+3PXyrn383JVeGa5m1A2evizpByy2ej03IJoX+v1WyWjuHyoeYB1O+Sa+HyM+zukB\nYPUs9refHnkz2d9ZfbaXZ/Nm1uZVT9tKdm2U28DXXqj1yvjw6ax3fepPTjD4F3xtJ6KOznkAgeuH\nKpEEN1DZJLcB9QucBgzAI8ne9dyune3Du/UI8gvn3MDHRY6m9J/t3Md0Hs3aYQCIPfgC2aaY/dh2\ncH84XElV87W8YfeRefNsy/D9eXCtrw+N8XAKc0ezZrnlWNbZuRpyANjSUkl2cYL7k9aUP66sPW4D\n2Zvmsy56wThui55pcANbA+tXcIDsqmk8dg8LTtya5brEJnHflt602ctTKNRlCiGEEEIIIUSB0URL\nCCGEEEIIIQqMJlpCCCGEEEIIUWD6pdFKjTDYeMqb6xx/11TjpWlKs2ZkRMzXYOU9jxOrpsPRaIVp\nHEqjvJ7X1R5sT3Gch62dvLYU8PVV7nncWFWAr2nocGJgFTsxdlztBQCcU76C7GMeP4vs23f6MRBW\nfuQXZH91O8dTijrrrE+q4nMAwPcvGf/G/zt+4ccmKyTpkgga57+5xvwb2/0YXsvrWZ8Qc3QC8Sh/\np3/sdGI5AWhr9rVoVEbC1wREy9hXUm5cKSd9JOL7QaaSU61cznGzrlrha11ilXzetKMVQ5Lt6Ahf\np1MxkTUbRTG+ZtGQunamudzmcn5mW4pZC7DHFTcBwObt/rH9nEwiv9ZuRISfzzanLWp0dIIAcEjF\nRrIf3cmxBFNOjJMDSrZ4ZTSmOCZRU5pjjZw6apmX54kmPs/2dtZ5FUfZ18ckfF3qT188iezfHHs9\n2Ytb5nh5Mon8cXWGHE5cGtj+X4NT5/gxyWY6sdzmfZJ1Al+68TKyJxX5/ccPF3IMu6VJbnumV3Is\nQQAYUczlfGzSY2TXdXLcttoTOKYfAPxz6wFcZoavSdgVMs51HIae1CM207s2qr3Gv1pu3M6wWJ/d\nCdPZhI1r8uHG52p12kkvnle9r3/3zpp56zG/hiJ7pnH/7N5zAEhn+b5OjbGvZDL+fU/PZc3oZkdv\ntfspHo/FD9vllVFV0vv4fk/SH4/NrWbt14odPDZ6rpV19fEqX6uX7mQ/d/06aX1t2NZO1oKlpo4h\n20ijJYQQQgghhBBvHzTREkIIIYQQQogCo4mWEEIIIYQQQhSYfmm0oqVpVB/65pryRSXrvDTNWdZz\n7OhkbdTMUtZ2uHFrAD+eUEOatQdh65BbnLhZrhZqXJy1LG6cLQBIOvoqdy1s2FrmqLPKfHSsmc/j\nXI+2jP99H22fTPZDZ/6I7E9M5ZhMAPD381gr8r4qjtPkxs36Rt25XhnTb3gzvs/2Rl//U0gyCWD3\n3DevX2PKjyfW0sLXyu521tk6S9Rtib9WuXQkrzsujrNOJZXxfSfZzudxV8K7yp5sSBlRRz+WcLRT\nLU1OvKuQ81SMbCP73GmvkF1sfH3ZvVvnkR0xXGo86l+jeImr4+I8O53YW+1Zf71zprnZO7a/48ie\nkA3R2bgxZOrS1WSv6+AYdwDQ5lyfqaU7yXbjkD2wi+9ZUBd+59We4TI7sn5THY9kerXdOFr1SV9f\n5uKudR8d9+9zSFg10QceWu3r3eKz2N9u3cha25qjWc+3bIevi/6v5jPIPmrserLfO/pFL89N248i\ne0nbFLLr2tnv65o57hYAlMcdjWklazz69Ba3ANq3IYPtXSsV6cyvMXXbkbjTZyRRGC22G0fLPc8e\nZ3xl6nxdqovNDuN73wudFXzfmzL+WMLt9xcnR3ppXKqqWDfX8Ar3bdVr2B8b5vhj5gOc2Fubd7AO\nyh1/AcDuTq5/8Xbu20YtZx/edpzvF9W1rBd71RlDTyjy9WRPNnDs2fjW3WQPZremX7SEEEIIIYQQ\nosBooiWEEEIIIYQQBUYTLSGEEEIIIYQoMJpoCSGEEEIIIUSB6d9mGJsNKr/1ppDtXz79GS/NeQtY\neHtNzUtkH7D4w2TbFb5A+4VLeTOIK7YfQ/bIOG8aAPjBhd3AwZVR3iRhfBEL4QA/oFnW2QbB3SwD\nAFLGCf7qiEDHxTk4m1uvMNryBB0EgF/Mmkn2yCdYrLzqehZej/71U72WZ21nr5+/VUw8i9jYN+/B\n+0Y976WJz2Hx5UuNHMhuSx0H0Yzt9K9lZwMfc/XDNuYLK93LnS1yRMlRJ088JHBkhRMctpyFpofX\nbPLyfKXmPrIrnODCl676INluUEIAqEqwX3ek+ZEui/vB/pocMWpjo/MMOtesPRPis7b/gcj3Ne5+\nEruzvvg8afm7zomz2Hd3sbOjBoDVSQ7smHUE/u5mGYdU+L7wWssEst0NNGYkuB4AsCbJ4uWiKLdf\nbpu4M+lvQPO5wx8ie1uaNzRoSPFGRIB/HYcFBdikYUZNvXcs5WxesK2ON6H4w+m/JXtCzN+c5MxH\nP0t2+Xh+5r/wMLcjADBxCgcgnlDC/ZTrO1t2sl8AwGfmP0r2XXt406awK2bd6zicN7/oJxMf9dvy\ntgt50yt3Uwp3c4ww3A1wIn0IYJx13tG7mwg1ZbityezhzaFCybMZyHClw9mHpqHDHzO3pNgP7tm1\nIG+5za08Vi2ayfdol7OR3chKP+h0WYzHjdEY38POtN9ZjEtwG/baBN6IrT7BeRI1LV4Ze5y6r2nj\n4MMVUX980tTBeUYVZl+YPqFftIQQQgghhBCiwGiiJYQQQgghhBAFRhMtIYQQQgghhCgw/Vtt35aE\nff7VN8zZl/pJXnHss+adT/bUZUvJXn3tIq+MYsOLJ7d3OGtFQzRabnBhF1c75eomgPB1nd0JC5Sc\nrx6u5qMp7Qebqyhlnc1FS/jCjsXrec+761gOkjoavWuy9jZFa9sx7YI3veObl37US7PwU6znWzhm\nA9m1k1n/F6Z3W9rMuq7NrawtaE/5eSqKee17SYzXDFcX89rkiQlf3+fi+sqfXzrCS7Pu2weQnbiP\nv38kvZHLPI+DjALAOd++new7dvDa7KKQgMVucMOqkfz9OlLcLIRptKLV7sJxL8l+R7qMv/eoqP88\n18RYq1Lk6BU6rd9kTiriZ++VVg6eWBLhdeyTilgfAwDJUr7G65N8fcfEfI1DcxG3aensJLJLnfXz\nI4r99u2A4q1kr+lk3VdomyhZjR9oF8irN1q1xQ92vfmVqWTXbGR/+9Kk88huS/oB7yteZD94asI0\nsm857edenotv+DzZt09kf0ts4v5x/HN+QPv/vfBksufsZL8ODQCa6b2fHs7kC9gbe+gF79iGFGv6\n3PGHG1g42gf9lUvG5g+UHHX6lA+MWEb2nTg2/4mM897fylcAoH2q/+y57E7yuPLZdg5AjpB7WFTE\n1zfiaMQ7Z/C4NIyKmNM/OH7QkfTHDq0ZbltGjGENVqaa/aA84WsTdzTwnCAWccfdfj9dW8n99FZn\nn4PECi9LwdAvWkIIIYQQQghRYDTREkIIIYQQQogCo4mWEEIIIYQQQhSY/kdEiXRb85vNv4Y2s2xl\nr5+PWOnP9SJOIJ/RxbyGsyHlxxFoSvEa1ZIor2uNOWuXXZ0K4K9vdtNE4a9vdtNkvbWwJXk+B1qz\nvO6+td3Xj7mYWP9uXej67z7cv8Fi1B98DdmaP7DtfselJ5xC9rajfL3CpFNZ13XRpGfIPiTBnwNA\nfdeou5EAAAn5SURBVIZjBb3YVkv2rjTHTrppMcd1A4Apd7NvFN/9HNmz4ccNc8knfRnx+Drv2H2N\nB5I9JsHPSn3Sf1Yq4ryuOpXgdfylFRzn4tkXZnllzGp8xju2v5Nx4gu91OHHlRoVZf3nmtRIssN0\nmm6smrIon6cyxmvd69O8vhzwfWxMEd/HiTFfF/jXBtb9bdjFdZ09mmNvdWb8ursaj7okx6qbWLzL\nyxPSdA4/BhD/6Z1zlnnHVoxn3daum1hnN+oH7KOlo/2+Yedctrcs5jIufP5zXp6Y0/QfPXcN2Rec\nyM/3lxIf8coofYH1F60Hsl28njWmAHwdjnhLuGOSjPPuPGPZXxLGj5fpxsTqC27srdIIt3kb03sx\nSNEQZ+b0bWQ3hvRbrv5oWb0T2zHjjzvzNWGlpXxPkyl/zOmOs10iUX/M7I6Bi5zGaNceHtdVlubX\nir22czzZkxN+v3XB2GfJ/tZc1shPuDPvaQaMWj0hhBBCCCGEKDCaaAkhhBBCCCFEgdFESwghhBBC\nCCEKjCZaQgghhBBCCFFg+r8ZRr4NFJxAjqaIxZi2gwV2Y3/+pFdE9D94/ndIGW9gEBa8syrCInZ3\ng4k2y7YbyA8AUk6Qs74E6nPzlDmiUFecWp/mjRcAYHacReslT/sbGLh4m1vsw40tBkLYZh42ne7V\ndgM2TnoopOCr2fwzanq1+wZf61no/0YQptjfuCPvWZ1nJbN9h5em6TjH9lK0ekdcilHP53E+n4Ut\nect4OxCJ5t/AIOWIvN0An+7zDQDbUxwUuyPLvr2ujTeYOHR0nVfGPTvmkz3C2bCkYqQvOnY3zDhg\nzHayY04AyqKoLyouMvyMzSrhMnalfeF1tD1/uzjkiYQEr8/TBpfFfN/58ETeFOgHZ59B9urD+fpf\nfOxir4yWNLctd65mX3rHtFVengtH83kThv1rWQcHfp+zqM4r49IJT5B95a8vInvC3V4WmCg/XzZ/\nLFbRC81ZDlZd6gRHD9vAqxC445oiZyOx25oOH5TzDkeqE9yH7+lMeGnml3EfXeG0NQ/u5o0fACCb\n5XvoxmAvLWZfqij2269VzWP4gDNmTnX67eS63bwBk3F2V4o5m2M0tfGGcgAQK+J+a1IFbxblbmwH\nAH+t582jRq4MDak+KOgXLSGEEEIIIYQoMJpoCSGEEEIIIUSB0URLCCGEEEIIIQpM/zVa+XCioLk6\nk74w+7GLyT5xGgdTfLme148DQNTRI7jrPqN9iLJZFuc1qWlHr5HJ+vPSlHPMDcbWmeY1qh0pP5Df\nvVUcdLbmWl+35mHzrL12F9wOIMDmYOLqr4Y6A3kORGEpe57Xeo86ts1L05bl53NMlNfHP9XqB2/e\nleJgw+VOwOLdTjD1tR0cTBIAGtpYi5OoYPFKmF50bBEHll7fNorsTc1VZNeU+drWq5ecSfY7Z3JQ\n3YNKN3l5yrbuX23JvsBEfe2BzaPRcu8PAPxt+SFkR+rYV/7rvJvIDgt2ncpyoOrrF3Lk98asr7O7\nfgeLO6vi/Cw8U19LdtEP+RwA8N3PvZNsMzhyoGGD61OuP4XpmisirOV0tVMDoS9jpSJw/+1q3iui\nXC/A90HRN156ZA7ZZ5zxvJdmbSvrgFf9ljVZX/vK7V6eVe3cD5VG/WDW9HnE//zEstfJ/nH0VLJH\nFfl97PQS1oRvT3GblnbG1JEQfxxbxH3ZL18+gex3LVzi5dkY5/Z3TSmfx1eCFQ79oiWEEEIIIYQQ\nBUYTLSGEEEIIIYQoMJpoCSGEEEIIIUSBKbxGqwBMu+AVsjc4n4/Cyr1SD/fihF2s/kdH8hmQ4iGf\n5mo/02QJsa8JCYHl8UAL6yWvv/tksldd/Asvz79uYL1LfQfHwRtTzPGu2rIcWxAAvjbrXrKXtbMO\ndUVqrJenKc2rypfv4DX3s0Y3kP3Bcc96ZTQfP53sTY/zWv95pX4MteQoxdHKq5EFvFhbLan8vcXk\nB1gHcfWmC8muOGerl6cqwfHR7t/M+oyW5/meAoAbZqb2HXVkuzqJze/zNWklz3I8nDF9iEtjMxJy\n9Uhe3bX/XjxpWVPa4WhM/XhCvg/GnVh6zZbblVRIzNHiCGtI3fPs6HTjhfbhvvflmRqGZGpZ7zan\ndJuXZnXzQWSPvvklsq857XQvT+WDfJ8rNrMflKzn2FTZVeu8Mh5Kc7y0PR9iPdnakK5i1SrWPUd3\nOzouJ9ZeqtrX9zXN4LrPuXMF2S/fP8XLc/drHF9w9irulwdzxKxftIQQQgghhBCiwGiiJYQQQggh\nhBAFRhMtIYQQQgghhCgw+6VGSwghCk2HE8aowviakq9Wv0b2419LkD03+Skvz/KP/Zzs3zXVkN2U\n4Thb8xKbvTIWFDWSfc8uXnO/rt3X2cwtY73ON+azzmu3c95rrvqQV0Ylnib7rzMeIPvPLZVenvYa\naSlsNv+KfhPn7nV8iR/HbNx0joU2/n+ayL7tluPJ3v4c+xYA/OHD15D9SNtMsl8YV+vleamBNYA3\nz/ob2T/ZeSjZ28b4flC2iEWPLz18iJfGQzqcAWNTvcc5AoCjSleT7Wq4jnD0ogDQ5sTrSjhasNKI\nH/vzqSRrvUZFWWdTE2N9z6vg9kz0nVmXs/7ozpnHemnaJ7EmrjjJ2trpH3q53+ftPSpgOCNuejp/\non6eJ+zXoJGLey/jlcP8PLPwAtl7cxcD/aIlhBBCCCGEEAVGEy0hhBBCCCGEKDCaaAkhhBBCCCFE\ngdFESwghhBBCCCEKjDbDEEIMCyZ/50myP/7Qp700seVuePRdZE258km4nHnXh8leeQlvQlE2gTc8\nOLRmk1fGmkoOBvnUllqyW3ZzmQDw/MjJZMdvG0n2yOueItvd+CKMs055P9npKv+8M5/qv+B5yJEN\nkXAbjs5pO3iziA1H5Y+YvW1qLdm1Wccf436XffEr/052opGDyUaTfl1tLW/y8oEVl3OCl1/n9On8\ndTfIL7i3mYFI7IcHNp0/4LPLTVecRfaPF/C780iafTI5zj+HKeVjNsNlxHb4m2EU7+JyR61wgt3+\n3Q+Onhe7N7cn2H8xcQ5on21zAvq+ws8mABS/4hxwgqWbqB902iSc4NXus+m0Z2H3x3uenU2CTJHv\nO6ao9+/n1dWtR0hdbMrx65BNd9zzunV32+tCol+0hBBCCCGEEKLAaKIlhBBCCCGEEAVGEy0hhBBC\nCCGEKDDG9mNdrDGmHsD6wauO2IdMtdaOGazC5TtDHvmPGCjyHfFWkP+IgSLfEW+FPvlPvyZaQggh\nhBBCCCHyo6WDQgghhBBCCFFgNNESQgghhBBCiAKjiZYQQgghhBBCFBhNtIQQQgghhBCiwGiiJYQQ\nQgghhBAFRhMtIYQQQgghhCgwmmgJIYQQQgghRIHRREsIIYQQQgghCowmWkIIIYQQQghRYP4/rLot\nT01e+soAAAAASUVORK5CYII=\n",
   "text/plain": "<matplotlib.figure.Figure at 0x11a93cd68>"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

`FashionMNIST` is a subclass of `gluon.data.Dataset`, which defines how to get the `i`-th example. In order to use it in training, we need to get a (randomized) batch of examples. It can be easily done by `gluon.data.DataLoader`.

```{.python .input  n=4}
batch_size = 256
train_data = gluon.data.DataLoader(
    mnist_train, batch_size, shuffle=True)
```

The returned `train_data` is an iterator that yields batches of data and labels pairs.

```{.python .input  n=5}
for data, label in train_data:
    print(data.shape, label.shape)
    break
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(256, 28, 28, 1) (256,)\n"
 }
]
```

Finally, we create a validation dataset and data loader.

```{.python .input  n=6}
mnist_valid = gluon.data.vision.FashionMNIST(train=False)
valid_data = gluon.data.DataLoader(
    mnist_valid, batch_size, shuffle=False)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Downloading /Users/markhama/.mxnet/datasets/fashion-mnist/t10k-images-idx3-ubyte.gz from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/t10k-images-idx3-ubyte.gz...\nDownloading /Users/markhama/.mxnet/datasets/fashion-mnist/t10k-labels-idx1-ubyte.gz from https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/fashion-mnist/t10k-labels-idx1-ubyte.gz...\n"
 }
]
```

## Define the model

We reimplement the same LeNet introduced before. One difference here is that we changed the weight initialization method to `Xavier`, which is a popular choice for deep convolutional neural networks.

```{.python .input  n=7}
net = nn.Sequential()
with net.name_scope():
    net.add(
        nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(10)
    )
net.initialize(init=init.Xavier())
```

Besides the neural network, we need to define the loss function and optimization method for training. We will use standard softmax cross entropy loss for classification problems. It first performs softmax on the output to obtain the predicted probability, and then compares the label with the cross entropy.

```{.python .input  n=8}
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

The optimization method we picked is the standard stochastic gradient descent with constant learning rate of 0.1.

```{.python .input  n=9}
trainer = gluon.Trainer(net.collect_params(),
                        'sgd', {'learning_rate': 0.1})
```

The `trainer` is created with all parameters (both weights and gradients) in `net`. Later on, we only need to call the `step` method to update its weights.

## Train

We create two auxiliary functions before training. Even though both are supported by built-in functions in Gluon, we found it's straightforward to implement them from scratch. One calculates the model accuracy, the other transforms the data into `(batch, channel, height, weight)` format and normalize its values.

```{.python .input  n=10}
def acc(output, label):
    # output: (batch, num_output) ndarray
    # label: (batch, ) ndarray
    return (output.argmax(axis=1)==label).mean().asscalar()

def transform(data, label):
    # data: (batch, height, weight, channel) ndarray
    # label: (batch, ) ndarray
    return (data.transpose((0,3,1,2)).astype('float32')/255,
            label.astype('float32'))
```

Now we can implement the complete training loop.

```{.python .input  n=10}
for epoch in range(10):
    train_loss, train_acc, valid_acc = 0., 0., 0.
    tic = time()
    for data, label in train_data:
        data, label = transform(data, label)
        # forward + backward
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate traing metrics
        train_loss += loss.mean().asscalar()
        train_acc += acc(output, label)

    # calculate validation accuracy
    for data, label in valid_data:
        data, label = transform(data, label)
        valid_acc += acc(net(data), label)

    print("Epoch %d: Loss: %.3f, Train acc %.3f, Test acc %.3f,\
Time %.1f sec" % (
        epoch, train_loss/len(train_data),
        train_acc/len(train_data),
        valid_acc/len(valid_data), time()-tic))
```

```{.json .output n=None}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0: Loss: 1.041, Train acc 0.618, Test acc 0.773,Time 13.0 sec\nEpoch 1: Loss: 0.578, Train acc 0.779, Test acc 0.815,Time 12.5 sec\nEpoch 2: Loss: 0.484, Train acc 0.820, Test acc 0.836,Time 14.0 sec\nEpoch 3: Loss: 0.436, Train acc 0.839, Test acc 0.856,Time 12.4 sec\nEpoch 4: Loss: 0.401, Train acc 0.853, Test acc 0.858,Time 12.7 sec\n"
 }
]
```

## Save the model

Finally, we save the trained parameters onto disk, so that we can use them later.

```{.python .input}
net.save_params('net.params')
```
