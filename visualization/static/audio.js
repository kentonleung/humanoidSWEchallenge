class AudioEngine {
    constructor() {
        this.ctx = null;
        this.masterGain = null;
        
        this.bgmOscillators = [];
        this.bgmInterval = null;
        this.isMuted = false;
        this.isPlayingBGM = false;
        this.initialized = false;
    }

    init() {
        if (this.initialized) return;
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (!AudioContext) {
            console.warn("Web Audio API not supported in this browser");
            return;
        }
        this.ctx = new AudioContext();
        this.masterGain = this.ctx.createGain();
        this.masterGain.gain.value = 1.0; // Increased volume
        this.masterGain.connect(this.ctx.destination);
        this.initialized = true;
    }

    resume() {
        this.init();
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    toggleMute() {
        this.resume();
        this.isMuted = !this.isMuted;
        if (this.masterGain) {
            this.masterGain.gain.setTargetAtTime(this.isMuted ? 0 : 1.0, this.ctx.currentTime, 0.1);
        }
        return this.isMuted;
    }

    // --- Sound Effects ---

    playFootstep() {
        if (this.isMuted || !this.initialized) return;
        this.resume();
        const t = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(150, t);
        osc.frequency.exponentialRampToValueAtTime(40, t + 0.1);
        
        gain.gain.setValueAtTime(0.5, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.1);
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        osc.start(t);
        osc.stop(t + 0.15);
    }

    playPickup() {
        if (this.isMuted || !this.initialized) return;
        this.resume();
        const t = this.ctx.currentTime;
        
        // Shiny gem pickup (two high notes)
        [880, 1108].forEach((freq, i) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            
            osc.type = 'sine';
            osc.frequency.setValueAtTime(freq, t + i * 0.1);
            
            gain.gain.setValueAtTime(0, t + i * 0.1);
            gain.gain.linearRampToValueAtTime(0.3, t + i * 0.1 + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.01, t + i * 0.1 + 0.3);
            
            osc.connect(gain);
            gain.connect(this.masterGain);
            
            osc.start(t + i * 0.1);
            osc.stop(t + i * 0.1 + 0.35);
        });
    }

    playUnlock() {
        if (this.isMuted || !this.initialized) return;
        this.resume();
        const t = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        
        // Metallic clunk
        osc.type = 'square';
        osc.frequency.setValueAtTime(300, t);
        osc.frequency.exponentialRampToValueAtTime(50, t + 0.15);
        
        gain.gain.setValueAtTime(0.4, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.15);
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        osc.start(t);
        osc.stop(t + 0.2);
    }

    playSuccess() {
        if (this.isMuted || !this.initialized) return;
        this.resume();
        const t = this.ctx.currentTime;
        // Triumphant Fanfare: C E G C
        const notes = [261.63, 329.63, 392.00, 523.25];
        
        notes.forEach((freq, i) => {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            
            osc.type = 'square';
            const startTime = t + i * 0.15;
            osc.frequency.setValueAtTime(freq, startTime);
            
            gain.gain.setValueAtTime(0, startTime);
            gain.gain.linearRampToValueAtTime(0.2, startTime + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.01, startTime + (i === 3 ? 0.8 : 0.15));
            
            osc.connect(gain);
            gain.connect(this.masterGain);
            
            osc.start(startTime);
            osc.stop(startTime + (i === 3 ? 0.8 : 0.15));
        });
    }

    playError() {
        if (this.isMuted || !this.initialized) return;
        this.resume();
        const t = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(100, t);
        osc.frequency.linearRampToValueAtTime(80, t + 0.2);
        
        gain.gain.setValueAtTime(0.3, t);
        gain.gain.linearRampToValueAtTime(0.01, t + 0.2);
        
        osc.connect(gain);
        gain.connect(this.masterGain);
        
        osc.start(t);
        osc.stop(t + 0.25);
    }

    playWeaponPickup() {
        if (this.isMuted || !this.initialized) return;
        this.resume();
        const t = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(400, t);
        osc.frequency.linearRampToValueAtTime(1200, t + 0.1);
        gain.gain.setValueAtTime(0, t);
        gain.gain.linearRampToValueAtTime(0.3, t + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.4);
        osc.connect(gain);
        gain.connect(this.masterGain);
        osc.start(t);
        osc.stop(t + 0.5);
    }

    playCombat() {
        if (this.isMuted || !this.initialized) return;
        this.resume();
        const t = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'square';
        osc.frequency.setValueAtTime(200, t);
        osc.frequency.exponentialRampToValueAtTime(20, t + 0.4);
        gain.gain.setValueAtTime(0.5, t);
        gain.gain.exponentialRampToValueAtTime(0.01, t + 0.4);
        osc.connect(gain);
        gain.connect(this.masterGain);
        osc.start(t);
        osc.stop(t + 0.5);
    }

    playAgentDeath() {
        if (this.isMuted || !this.initialized) return;
        this.resume();
        const t = this.ctx.currentTime;
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(150, t);
        osc.frequency.linearRampToValueAtTime(40, t + 0.8);
        gain.gain.setValueAtTime(0.5, t);
        gain.gain.linearRampToValueAtTime(0, t + 0.8);
        osc.connect(gain);
        gain.connect(this.masterGain);
        osc.start(t);
        osc.stop(t + 1.0);
    }

    // --- Background Music ---
    
    startBGM() {
        this.resume();
        if (this.isPlayingBGM || this.isMuted || !this.initialized) return;
        this.isPlayingBGM = true;
        
        // Dark fantasy arpeggio (Am -> F -> Dm -> E)
        const chords = [
            [220.00, 261.63, 329.63], // Am
            [174.61, 220.00, 261.63], // F
            [146.83, 174.61, 220.00], // Dm
            [164.81, 207.65, 246.94]  // E
        ];
        
        let chordIndex = 0;
        let noteIndex = 0;
        
        this.bgmInterval = setInterval(() => {
            if (this.isMuted || !this.initialized) return;
            const t = this.ctx.currentTime;
            
            const chord = chords[chordIndex];
            const freq = chord[noteIndex];
            
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            
            // Soft triangle wave for retro RPG vibe
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(freq, t);
            
            // Pluck envelope
            gain.gain.setValueAtTime(0, t);
            gain.gain.linearRampToValueAtTime(0.1, t + 0.05);
            gain.gain.exponentialRampToValueAtTime(0.01, t + 0.4);
            
            osc.connect(gain);
            gain.connect(this.masterGain);
            
            osc.start(t);
            osc.stop(t + 0.5);
            
            noteIndex++;
            if (noteIndex >= chord.length) {
                noteIndex = 0;
                chordIndex = (chordIndex + 1) % chords.length;
            }
        }, 300); // 300ms per note
    }
    
    stopBGM() {
        this.isPlayingBGM = false;
        if (this.bgmInterval) {
            clearInterval(this.bgmInterval);
            this.bgmInterval = null;
        }
    }
}

// Global instance
window.audioEngine = new AudioEngine();
