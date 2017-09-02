�
#��Yc           @   s   d  d l  Z d �  Z d S(   i����Nc   
      C   s�   |  j  d d � }  | j  d d � } t j t j |  � � } t j t j | � � } t j t j t j | | f � � � } |  | }  | | } t j |  � } t j | � } t j | | � } t j	 t
 | d � t
 | d � � } | | }	 |	 S(   sY  
    Arguments:
    ---------------------------
    x: 1D or 2D numpy array
    y: 1D or 2D numpy array (dims must agree with x)

    Note, if there are Nan in either vector, the corresponding indices will be
    removed in each array

    Returns:
    ---------------------------
    c: The pearson correlation coefficient between x and y

    i����i   i   (   t   reshapet   npt   wheret   isnant   sortt   uniquet   concatenatet   ediff1dt   dott   sqrtt   sum(
   t   xt   yt   x_indst   y_indst   nan_indst   x_diffst   y_diffst   numt   dent   c(    (    s[   /home/charlie/Desktop/swdb_2017_tools/swdb2017/brain_observatory/util/pearson_corr_coeff.pyt   pearson_corr_coeff   s    '

'
(   t   numpyR   R   (    (    (    s[   /home/charlie/Desktop/swdb_2017_tools/swdb2017/brain_observatory/util/pearson_corr_coeff.pyt   <module>   s   
