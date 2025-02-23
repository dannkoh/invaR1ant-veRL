(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in8 Int)
(declare-const in7 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and (and (and (and (and (and  ( =  in0 120)  ( =  in1 120))  ( =  in2 120))  ( =  in3 120))  ( =  in4 120))  ( =  in5 120))  ( =  in6 120))  ( =  in7 120))  ( =  in8 120)))

(check-sat)
(get-model)