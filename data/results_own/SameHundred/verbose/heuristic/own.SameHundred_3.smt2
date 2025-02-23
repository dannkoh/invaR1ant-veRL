(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)

(assert (and (and  ( <  in0 100)  ( <  in1 100))  ( <  in2 100)))

(check-sat)
(get-model)