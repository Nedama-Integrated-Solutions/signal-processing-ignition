package com.nedama.signal.common81;

import com.inductiveautomation.ignition.common.script.ScriptManager;
import com.inductiveautomation.ignition.common.script.hints.PropertiesFileDocProvider;

/** Registers the script module. Every scope's hook calls this. */
public final class SignalRegistration {

    /** Scripting namespace. */
    public static final String SCRIPT_PATH = "system.signal";

    private SignalRegistration() {
    }

    public static void registerScripts(ScriptManager manager) {
        manager.addScriptModule(SCRIPT_PATH, new SignalScriptModule(), new PropertiesFileDocProvider());
    }
}
