(declare-const in0 Int)
(declare-const in1 Int)

(assert (and (and (and  ( >=  in0 97)  ( <=  in0 122))  ( >=  in1 97))  ( <=  in1 122)))

(check-sat)
(get-model)