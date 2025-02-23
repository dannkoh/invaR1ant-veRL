(declare-const in3 Int)

(assert  ( =  in3 0))

(check-sat)
(get-model)