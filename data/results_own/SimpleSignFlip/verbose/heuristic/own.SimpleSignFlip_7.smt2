(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and  ( =  in0 in1)  ( <  in1 in2))  ( =  in2 in3))  ( <  in3 in4))  ( =  in4 in5))  ( <  in5 in6)))

(check-sat)
(get-model)