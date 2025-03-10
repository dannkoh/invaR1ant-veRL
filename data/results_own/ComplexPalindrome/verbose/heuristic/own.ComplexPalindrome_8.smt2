(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in7 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)
(declare-const in3 Int)

(assert (and (and (and  ( <  in0 in7)  ( <  in1 in6))  ( <  in2 in5))  ( <  in3 in4)))

(check-sat)
(get-model)