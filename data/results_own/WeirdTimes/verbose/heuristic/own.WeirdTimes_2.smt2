(declare-const in0 Int)
(declare-const in1 Int)

(assert  ( =  in1 ( *  in0 2)))

(check-sat)
(get-model)