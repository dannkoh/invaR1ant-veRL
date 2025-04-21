(declare-const in6 Int)
(declare-const in5 Int)
(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in4 Int)

(assert (and (and  ( <  in0 in6)  ( <  in1 in5))  ( <  in2 in4)))

(check-sat)
(get-model)