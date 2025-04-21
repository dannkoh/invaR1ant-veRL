(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in7 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and  ( =  ( -  in2 in1) ( -  in1 in0))  ( =  ( -  in3 in2) ( -  in2 in1)))  ( =  ( -  in4 in3) ( -  in3 in2)))  ( =  ( -  in5 in4) ( -  in4 in3)))  ( =  ( -  in6 in5) ( -  in5 in4)))  ( =  ( -  in7 in6) ( -  in6 in5))))

(check-sat)
(get-model)