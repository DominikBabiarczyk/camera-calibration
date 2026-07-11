---
description: "Advice to use `#tool:vscode/askQuestions` when performing any task"
applyTo: "**"
---

* Zawsze używaj narzędzia pytań `#tool:vscode/askQuestions` do zadawania pytań, zamiast prosić o odpowiedź w zwykłej wiadomości.
* Jeśli wykryjesz problem designowy, architektoniczny albo niespójność, opisz go konkretnie, wskaż ryzyko lub koszt i zadaj użytkownikowi pytanie, zanim pójdziesz dalej w niepewnym kierunku.
* Nie zgaduj w miejscach, które wpływają na API, zachowanie programu, format danych, nazewnictwo publiczne, architekturę albo sposób testowania.
* Po każdej rundzie poprawek użyj `#tool:vscode/askQuestions` i zapytaj co poprawić.
* Kontynuuj ten cykl, dopóki użytkownik jednoznacznie nie napisze w `#tool:vscode/askQuestions`, że efekt jest już OK. 
* NIGDY nie kończ pracy bez pytania o poprawki, chyba że użytkownik wyraźnie potwierdził, że nic więcej nie trzeba zmieniać w toolu `#tool:vscode/askQuestions`.
* Jeśli na zadane pytanie masz kilka propozycji, to przedstaw je wszystkie i pozwól użytkownikowi wybrać, ale zawsze pozwól też użytkownikowi zaproponować własne rozwiązanie.
* Nie zaczynaj pisania kodu, jeśli coś jest niejasne, a odpowiedź można uzyskać tylko od użytkownika. Zawsze zadawaj pytania, zamiast zgadywać.
