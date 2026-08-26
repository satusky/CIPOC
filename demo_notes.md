# Demo things that need to be fixed

1. The variable nodes inside the group container don't need individual lines to/from the gate. They don't add anything and make it too busy.
2. The note strip at the top doesn't need the `initialize_case` step. Just contain all the nodes in a single `scan notes` container The notes can then take a form similar to a group of variables, i.e. a container with the circles that fill as they are completed. No individual lines to or from anything.
3. I don't like the plan and update blocks in the case. I think that just having these lines flow to the overall case block is sufficient as long as the current process is reflected by the label (preferrably colored to match the lines that are showing the data flow from that step). Removing those blocks allows for much larger text in the case box to show the current step.
4. I think we can also remove the finalize case block and just have the central case block say finalizing or something similar.

