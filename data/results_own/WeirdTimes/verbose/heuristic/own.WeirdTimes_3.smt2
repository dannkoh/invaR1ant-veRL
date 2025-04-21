(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)

(assert (and  ( =  in1 ( *  in0 2))  ( =  in2 ( *  in0 3))))

(check-sat)
(get-model)