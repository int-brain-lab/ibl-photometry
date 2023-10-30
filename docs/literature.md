---
title: Photometry - technical overview
author:
    - name: Kcénia Bougrova
    - name: Olivier Winter
    - name: Romain Ligneul
format: docx
bibliography: references.bib
---

## Technical overview

This overview is focused on the technicalities of bulk photometry recordings. Some other overviews have been recently published that may be consulted [@mejaes2022].

### Bleaching, decay and temporal drifts

Bulk fluorescence signals are unstable by nature. Even if the bleaching of GcAMP is limited compared with 2-photon imaging that uses much strong light beams, it still contributes to signal decay (possibly starting around [40uW at patchcord tip](https://blog.mohebial.com/led-power-tuning-for-fiber-photometry/)). Other sources of signal decay are less well understood, but they may involve all or part of the following mechanisms:

-   **Reduction in apparatus autofluorescence**. Typically, the patchcords used for photometric recordings are autofluorescent (even the "low autofluorescence" versions tailored to photometry). This type of autofluorescence is mostly driven by the connection between different elements of the recording apparatuses, and it tends to decrease over time. Recordings without any connection to a living animal can highlight this phenomenon.

-   **Reduction in GcAMP-independent tissue autofluorescence**. Brain tissue is intrinsically autofluorescent and this autofluorescence is also susceptible to bleaching. This phenomenon can typically be studied by comparing signal decay when no animal is connected to the apparatus with the decay obtained with an animal that *does not* express any fluorophore is connected to the apparatus.

-   **Green-to-red photoconversion of GCaMP**. This phenomenon is less obvious and harder to characterize but it has been reported for various versions of GCaMP [@ai2015] and more generally for GFP proteins [@gorbachev2020]. It may be studied with a recording system like our in-house Doric apparatus which has the capability to record red light (it should lead to an increase in the red-to-green signal ratio over time). Apparently the phenomenon is exacerbated in low oxygen environments.

-   **Reduction in light power over time**. This issue is less common but it can arise when LED cells are "burnt" over the course of an experiment. With old lasers, the temporal drift can be unpredictable (which is to be avoided at all cost!)

Signal decay can unfortunately happen at different rates for different excitation frequencies (e.g. 405, 470, 565, etc.), which generally requires to correct temporal drifts on a *per* channel basis before attempting movement artifact correction.

### Isosbestic signal and other movement correction methods

One key problem with bulk photometry recordings is that they can be significantly artifacted by movements. Movement can lead to translations of the brain tissue underneath the implanted fiber *and* changes in the alignment between the output of the implanted fiber and the patchcord that transmits light to the recording apparatus. For this reason, neuroscientists using photometry routinely use movement correction. There are 3 main methods that can be used separately or in conjunction:

-   **Baseline correction**. The signal is corrected by a substraction or a division of a baseline signal (either continuous or time-locked to a specific event). One possible approach is to correct using the mean signal recorded in a time window (-2 to -1 seconds) *before* every data point. This approach is only useful to "flatten" the fluorescence trace, but it does not prevent contamination of results by artifacts "aligned" with the events of interest (e.g. the artifacts induced by licking a water spout, or those induced by locomotor differences associated with different conditions).

-   **Red channel correction**. The signal is corrected by *regressing out* the fluorescence produced by a second fluorophore (generally expressed after a viral injection and emitting in a different wavelength, such as the (red) tdTomato protein) to correct the (green) signal of interest. This technique is view by some as a gold standard, but it also has important downsides:

    -   The second fluorophore is usually not expressed in the exact same set of neurons than GCaMP (except when using viruses coding for both at the same time, but these tend to work poorly), which can lead to over- or under-correction of movement artifacts

    -   The emission light of the second fluorophore can leak into the channel of interest, especially if the emission spectrum is broad and that the optical filters of the recording system are permissive.

    -   Expressing the second fluorophore requires a second virus, or a transgenic line expressing either this fluorophore or GCaMP natively.

-   **Isosbestic correction**. Isosbestic signals are (in theory) *calcium-independent* signals produced by GCaMP proteins when using excitation lights in the 405-415 wavelength range. Typically, isosbestic correction are obtained by interleaving two excitation wavelengths in time (an other method is to stimulate both fluorophore using decorrelated sinusoidal light intensities), so as to obtain calcium-dependent and calcium-independent signals. After temporal interpolation, the calcium-independent signal can be used to *regress out* movement artifacts, exactly as would be done if it originated from a second fluorophore. One key advantage of this technique is that it does not require to express another fluorophore and that it can be deemed *ratiometric* in the sense that both signals come from the exact same population of neurons. However, between theory and practice, there are a couple of issues to pinpoint:

    -   A 410nm light beam does not penetrate brain tissue exactly like 470nm light beam does, and it does not induce the exact same autofluorescence levels. In practice, and given that its apparent signal to noise ratio is lower, isosbestic signals may therefore be suspected of under-correcting movement artifacts.

    -   In high-quality recordings, isosbestic signals are often *anticorrelated* with GCaMP signals of interest. While in theory (note that our understanding of GCAMP fluorescence chemistry is still limited), the isosbestic signal should reflect the *total* amount of GCaMP (both calcium-bounded and free), in practice it is not so simple [@barnett2017] and anticorrelation are observed [@formozov2023]. The literature on this issue is messy and the isosbestic signal has been used in many different ways over the years. Another likely source of contamination of the isosbestic signal is hemodynamic mechanisms (see below).

### Hemodynamic confounds

Unfortunately, hemoglobin is fluorescent and oxygenated hemoglobin absorbs green light (510-560nm). That's useful to measure heart-beat in humans, but it is very inconvenient to measure GCaMP responses [@valley2020; @zhang2022], especially when it comes to highly vascularized structures such as the brainstem. This effect likely contributes to the slow time-course of some fluorescence traces observed experimentally. It is a

### Normalization across animals and sessions

There are two main ways of normalizing signals across animals and sessions. By computing a DF/F0 metric or simply by z-scoring the traces.

### Filtering

### Origins of the signal

### References

::: {#refs}
:::