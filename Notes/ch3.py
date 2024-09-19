Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
from sklearn.datasets import fetch_openml
>>> mnist = fetch_openml('mnist_784', version=1)
>>> mnist.keys()
SyntaxError: multiple statements found while compiling a single statement
from sklearn.datasets import fetch_openml

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)



dataset = fetch_openml('mnist_784', version=1)
dataset.keys()
dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])
dataset.DESCR
"**Author**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges  \n**Source**: [MNIST Website](http://yann.lecun.com/exdb/mnist/) - Date unknown  \n**Please cite**:  \n\nThe MNIST database of handwritten digits with 784 features, raw data available at: http://yann.lecun.com/exdb/mnist/. It can be split in a training set of the first 60,000 examples, and a test set of 10,000 examples  \n\nIt is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.  \n\nWith some classification methods (particularly template-based methods, such as SVM and K-nearest neighbors), the error rate improves when the digits are centered by bounding box rather than center of mass. If you do this kind of pre-processing, you should report it in your publications. The MNIST database was constructed from NIST's NIST originally designated SD-3 as their training set and SD-1 as their test set. However, SD-3 is much cleaner and easier to recognize than SD-1. The reason for this can be found on the fact that SD-3 was collected among Census Bureau employees, while SD-1 was collected among high-school students. Drawing sensible conclusions from learning experiments requires that the result be independent of the choice of training set and test among the complete set of samples. Therefore it was necessary to build a new database by mixing NIST's datasets.  \n\nThe MNIST training set is composed of 30,000 patterns from SD-3 and 30,000 patterns from SD-1. Our test set was composed of 5,000 patterns from SD-3 and 5,000 patterns from SD-1. The 60,000 pattern training set contained examples from approximately 250 writers. We made sure that the sets of writers of the training set and test set were disjoint. SD-1 contains 58,527 digit images written by 500 different writers. In contrast to SD-3, where blocks of data from each writer appeared in sequence, the data in SD-1 is scrambled. Writer identities for SD-1 is available and we used this information to unscramble the writers. We then split SD-1 in two: characters written by the first 250 writers went into our new training set. The remaining 250 writers were placed in our test set. Thus we had two sets with nearly 30,000 examples each. The new training set was completed with enough examples from SD-3, starting at pattern # 0, to make a full set of 60,000 training patterns. Similarly, the new test set was completed with SD-3 examples starting at pattern # 35,000 to make a full set with 60,000 test patterns. Only a subset of 10,000 test images (5,000 from SD-1 and 5,000 from SD-3) is available on this site. The full 60,000 sample training set is available.\n\nDownloaded from openml.org."
target = dataset['target']
target.head()
0    5
1    0
2    4
3    1
4    9
Name: class, dtype: category
Categories (10, object): ['0', '1', '2', '3', ..., '6', '7', '8', '9']
target[0:1]
0    5
Name: class, dtype: category
Categories (10, object): ['0', '1', '2', '3', ..., '6', '7', '8', '9']
data = dataset['data']
type(data)
<class 'pandas.core.frame.DataFrame'>
data.head()
   pixel1  pixel2  pixel3  pixel4  ...  pixel781  pixel782  pixel783  pixel784
0       0       0       0       0  ...         0         0         0         0
1       0       0       0       0  ...         0         0         0         0
2       0       0       0       0  ...         0         0         0         0
3       0       0       0       0  ...         0         0         0         0
4       0       0       0       0  ...         0         0         0         0

[5 rows x 784 columns]
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 70000 entries, 0 to 69999
Columns: 784 entries, pixel1 to pixel784
dtypes: int64(784)
memory usage: 418.7 MB
data.head(1)
   pixel1  pixel2  pixel3  pixel4  ...  pixel781  pixel782  pixel783  pixel784
0       0       0       0       0  ...         0         0         0         0

[1 rows x 784 columns]
data.columns
Index(['pixel1', 'pixel2', 'pixel3', 'pixel4', 'pixel5', 'pixel6', 'pixel7',
       'pixel8', 'pixel9', 'pixel10',
       ...
       'pixel775', 'pixel776', 'pixel777', 'pixel778', 'pixel779', 'pixel780',
       'pixel781', 'pixel782', 'pixel783', 'pixel784'],
      dtype='object', length=784)
d = data[:1]
import numpy as np
d = np.array(data[:1])
type(d)
<class 'numpy.ndarray'>
d.shape
(1, 784)
d = d.reshape(28,28)
d.shape
(28, 28)
d
array([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   3,
         18,  18,  18, 126, 136, 175,  26, 166, 255, 247, 127,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,  30,  36,  94, 154, 170,
        253, 253, 253, 253, 253, 225, 172, 253, 242, 195,  64,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  49, 238, 253, 253, 253, 253,
        253, 253, 253, 253, 251,  93,  82,  82,  56,  39,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  18, 219, 253, 253, 253, 253,
        253, 198, 182, 247, 241,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,  80, 156, 107, 253, 253,
        205,  11,   0,  43, 154,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,  14,   1, 154, 253,
         90,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 139, 253,
        190,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  11, 190,
        253,  70,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
        241, 225, 160, 108,   1,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
         81, 240, 253, 253, 119,  25,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,  45, 186, 253, 253, 150,  27,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,  16,  93, 252, 253, 187,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0, 249, 253, 249,  64,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,  46, 130, 183, 253, 253, 207,   2,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  39,
        148, 229, 253, 253, 253, 250, 182,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  24, 114, 221,
        253, 253, 253, 253, 201,  78,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,  23,  66, 213, 253, 253,
        253, 253, 198,  81,   2,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,  18, 171, 219, 253, 253, 253, 253,
        195,  80,   9,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,  55, 172, 226, 253, 253, 253, 253, 244, 133,
         11,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0, 136, 253, 253, 253, 212, 135, 132,  16,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0]], dtype=int64)
np.set_printoptions(linewidth=50)
d
array([[  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   3,  18,  18,  18,
        126, 136, 175,  26, 166, 255, 247, 127,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
         30,  36,  94, 154, 170, 253, 253, 253,
        253, 253, 225, 172, 253, 242, 195,  64,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  49,
        238, 253, 253, 253, 253, 253, 253, 253,
        253, 251,  93,  82,  82,  56,  39,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,  18,
        219, 253, 253, 253, 253, 253, 198, 182,
        247, 241,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
         80, 156, 107, 253, 253, 205,  11,   0,
         43, 154,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,  14,   1, 154, 253,  90,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0, 139, 253, 190,   2,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,  11, 190, 253,  70,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,  35, 241, 225, 160,
        108,   1,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,  81, 240, 253,
        253, 119,  25,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,  45, 186,
        253, 253, 150,  27,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,  16,
         93, 252, 253, 187,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0, 249, 253, 249,  64,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,  46, 130,
        183, 253, 253, 207,   2,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,  39, 148, 229, 253,
        253, 253, 250, 182,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,  24, 114, 221, 253, 253, 253,
        253, 201,  78,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
         23,  66, 213, 253, 253, 253, 253, 198,
         81,   2,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,  18, 171,
        219, 253, 253, 253, 253, 195,  80,   9,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,  55, 172, 226, 253,
        253, 253, 253, 244, 133,  11,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0, 136, 253, 253, 253,
        212, 135, 132,  16,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0],
       [  0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0]], dtype=int64)
import pandas as pd
import matplotlib.pyplot as plt
plt.imshow()
Traceback (most recent call last):
  File "<pyshell#29>", line 1, in <module>
    plt.imshow()
TypeError: imshow() missing 1 required positional argument: 'X'
plt.imshow(d)
<matplotlib.image.AxesImage object at 0x000001A15A79F790>
plt.imshow(d, cmap = 'binary')
<matplotlib.image.AxesImage object at 0x000001A159B5BA30>
plt.axis('off')
(-0.5, 27.5, 27.5, -0.5)
plt.show()
plt.show()
e = np.array(data[1:2])
plt.imshow(d.reshape(28,28), cmap='binary')
<matplotlib.image.AxesImage object at 0x000001A15C0192A0>
plt.imshow(e.reshape(28,28), cmap='binary')
<matplotlib.image.AxesImage object at 0x000001A15C03D930>
plt.show()
mnist = fetch_openml('mnist_784', as_frame=False)

Warning (from warnings module):
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 109
    warn(
UserWarning: A network error occurred while downloading https://api.openml.org/api/v1/json/data/list/data_name/mnist_784/limit/2/status/active/. Retrying...
Traceback (most recent call last):
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 1348, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1283, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1329, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1278, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1038, in _send_output
    self.send(msg)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 976, in send
    self.connect()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1448, in connect
    super().connect()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 942, in connect
    self.sock = self._create_connection(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\socket.py", line 824, in create_connection
    for res in getaddrinfo(host, port, 0, SOCK_STREAM):
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\socket.py", line 955, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
socket.gaierror: [Errno 11001] getaddrinfo failed

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#39>", line 1, in <module>
    mnist = fetch_openml('mnist_784', as_frame=False)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\utils\_param_validation.py", line 213, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 1010, in fetch_openml
    data_info = _get_data_info_by_name(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 301, in _get_data_info_by_name
    json_data = _get_json_content_from_openml_api(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 245, in _get_json_content_from_openml_api
    return _load_json()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 66, in wrapper
    return f(*args, **kw)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 240, in _load_json
    _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 172, in _open_openml_url
    _retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 102, in wrapper
    return f(*args, **kwargs)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 216, in urlopen
    return opener.open(url, data, timeout)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 519, in open
    response = self._open(req, data)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 536, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 496, in _call_chain
    result = func(*args)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 1391, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 1351, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>
mnist = fetch_openml('mnist_784', as_frame=False)
Traceback (most recent call last):
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 1348, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1283, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1329, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1278, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1038, in _send_output
    self.send(msg)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 976, in send
    self.connect()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1448, in connect
    super().connect()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 942, in connect
    self.sock = self._create_connection(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\socket.py", line 824, in create_connection
    for res in getaddrinfo(host, port, 0, SOCK_STREAM):
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\socket.py", line 955, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
socket.gaierror: [Errno 11001] getaddrinfo failed

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#40>", line 1, in <module>
    mnist = fetch_openml('mnist_784', as_frame=False)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\utils\_param_validation.py", line 213, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 1010, in fetch_openml
    data_info = _get_data_info_by_name(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 301, in _get_data_info_by_name
    json_data = _get_json_content_from_openml_api(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 245, in _get_json_content_from_openml_api
    return _load_json()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 66, in wrapper
    return f(*args, **kw)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 240, in _load_json
    _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 172, in _open_openml_url
    _retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 102, in wrapper
    return f(*args, **kwargs)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 216, in urlopen
    return opener.open(url, data, timeout)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 519, in open
    response = self._open(req, data)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 536, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 496, in _call_chain
    result = func(*args)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 1391, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 1351, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>
mnist = fetch_openml('mnist_784')
Traceback (most recent call last):
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 1348, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1283, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1329, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1278, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1038, in _send_output
    self.send(msg)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 976, in send
    self.connect()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 1448, in connect
    super().connect()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\http\client.py", line 942, in connect
    self.sock = self._create_connection(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\socket.py", line 824, in create_connection
    for res in getaddrinfo(host, port, 0, SOCK_STREAM):
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\socket.py", line 955, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
socket.gaierror: [Errno 11001] getaddrinfo failed

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<pyshell#41>", line 1, in <module>
    mnist = fetch_openml('mnist_784')
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\utils\_param_validation.py", line 213, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 1010, in fetch_openml
    data_info = _get_data_info_by_name(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 301, in _get_data_info_by_name
    json_data = _get_json_content_from_openml_api(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 245, in _get_json_content_from_openml_api
    return _load_json()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 66, in wrapper
    return f(*args, **kw)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 240, in _load_json
    _open_openml_url(url, data_home, n_retries=n_retries, delay=delay)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 172, in _open_openml_url
    _retry_on_network_error(n_retries, delay, req.full_url)(urlopen)(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\datasets\_openml.py", line 102, in wrapper
    return f(*args, **kwargs)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 216, in urlopen
    return opener.open(url, data, timeout)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 519, in open
    response = self._open(req, data)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 536, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 496, in _call_chain
    result = func(*args)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 1391, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\urllib\request.py", line 1351, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [Errno 11001] getaddrinfo failed>
