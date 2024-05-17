# Participation: multy-agent-based simulation based on mesa

Codebase of a simulation for the master thesis 
"Influence of different voting rules on participation and welfare in a simulated multi-agent democracy" 
at the group [Swarm Intelligence and Complex Systems](https://siks.informatik.uni-leipzig.de) 
at the [Faculty of Mathematics and Computer Science](https://www.mathcs.uni-leipzig.de/en)
of [Leipzig University](https://www.uni-leipzig.de/en).

## Short overview in German

• Multi-Agenten Simulation 
  - untersucht werden soll die Partizipation der Agenten an den Wahlen
  - Auswirkung von verschiedenen Wahlverfahren auf die Partizipation
  - Verlauf der Partizipation über die Zeit (Umgebungsänderung, Änderung der Vermögensverteilung, ...)

• Umgebung:
  - Gitterstruktur ohne Ränder
  - jedes Viereck im Gitter ist ein Feld, eine Menge von (zusammenhängenden) Feldern ist ein Gebiet 
  - Jedes Feld im Gitter hat eine von vier Farben (r, g, b, w) diese verändert sich mit einer bestimmten 
    "Mutationsrate", die Änderung ist abhängig von den Ergebnissen der letzten Gruppenentscheidung 
    über ein Gebiet, welches das Feld einschließt (d.h. die Umgebung reagiert auf die Gruppenentscheidungen)

• Agenten 
  - Intelligenz: top-down approach d.h. die Agenten bekommen durch ein training eine einfache KI 
    (auf Basis von Entscheidungsbäumen um das Verhalten nachvollziehen zu können)
  - Haben bestimmtes Budget (Motivation)
  - Entscheidungen (Agenten können):
    - umliegende Felder erkunden ("sich bilden") - kostet
    - an Wahlen teilnehmen - kostet
    - abwarten - geringe kosten
  - "Agentenpersönlichkeit": jede Agentin besitzt zwei Präferenzrelationen über die Farben r, g und b
  - ⇨ 15 Agententypen (zufällig und normalverteilt)
    - diese wirken sich auf die Belohnungen der Abstimmungsergebnisse aus

• Wahlen
  - Abgestimmt wird über die Häufigkeitsverteilung der 4 Farben im Wahlgebiet (objektive Wahrheit soll 
    "kluge Gruppenentscheidung" simulieren)
  - Belohnung wird an alle Agenten im Wahlgebiet ausgeschüttet:
    - je näher (kendal-tau Distanz) die abgestimmte Verteilung (Wahlergebnis) an der wahren Verteilung liegt, 
      umso größer die Belohnung
    - eine Hälfte der Belohnung geht an alle zu gleichen Teilen
    - zweite Hälfte wird entsprechend der "Agentenpersönlichkeit" (siehe oben) ausgeschüttet
      
    - ⇨ abstimmende Agenten im Zwiespalt:
      - so abstimmen wie die Verteilung vermutlich wirklich ist (gute Entscheidung für alle - nach bestem wissen) 
      - oder eher egoistisch, sodass jetzt und in Zukunft möglichst viel an den Agenten selbst geht 
              (beachte: Das Ergebnis beeinflusst auch die zukünftige Verteilung im Gebiet)
    
Interessante Fragen:
- machen die verschiedenen Wahlverfahren einen Unterschied und wenn ja welchen?
- wie verhalten sich Agententypen die in der Minderheit/Mehrheit sind?
- wie wirkt sich (nicht) Partizipation langfristig aus?
- wie wirkt sich die Verteilung von Vermögen auf die Partizipation aus und umgekehrt? 
- welche Muster entstehen in den Gebieten (lokal, regional, global)?
Auch interessant:
- was passiert, wenn die Gruppe der abstimmenden Agenten zufällig ausgewählt wird 
(„Bürgerräte“ also kostenlose oder vergütete Teilnahme von x% aller Agenten)? 
(werden die Entscheidungen „besser“, wie verteilt sich der Wohlstand, …) 

## Citing Mesa

To cite Mesa in your publication, 
you can use the [CITATION.bib](https://github.com/projectmesa/mesa/blob/main/CITATION.bib).
