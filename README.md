# infra-vision-assessment-engine
Rail Track Segmentation &amp; Weather Assessment Using DINOv2 This project implements a functional prototype for automated rail track segmentation using the DINOv2 foundation model. It is designed to evaluate railway infrastructure and environmental hazards through a semantic segmentation pipeline.

``` mermaid
graph TD
    %% -- Actors --
    User((User))

    %% -- Frontend Layer --
    subgraph Frontend ["Presentation Layer"]
        UI["Web Dashboard<br/>(Streamlit)"]
    end

    %% -- Application Logic Layer --
    subgraph Logic ["Application Logic"]
        Engine["Weather Assessment Engine<br/>(Evaluator)"]
        Knowledge[("Scenario Database<br/>(CSVs)")]
    end

    %% -- AI Model Layer --
    subgraph AI_Core ["AI & Inference Layer"]
        direction TB
        Model["DINOv2 Backbone<br/>(Segmentation Head)"]
        Pre["Preprocessing Pipeline<br/>(Augmentation)"]
    end

    %% -- Relationships --
    User --> UI
    UI --> Engine
    Engine --> Knowledge
    Engine --> Model
    Model --> Pre
```
