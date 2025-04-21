(declare-const in0 Int)
(declare-const in2 Int)
(declare-const in1 Int)
(declare-const in3 Int)

(assert (and (and  ( =  in1 ( *  in0 2))  ( =  in2 ( *  in0 3)))  ( =  in3 ( *  in0 4))))

(check-sat)
(get-model)