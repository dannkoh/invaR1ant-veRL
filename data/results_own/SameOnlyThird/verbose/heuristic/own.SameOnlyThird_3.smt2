(declare-const in2 Int)

(assert  ( =  in2 0))

(check-sat)
(get-model)